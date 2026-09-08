import type { Message } from "@langchain/langgraph-sdk";
import { afterEach, beforeEach, expect, rs, test } from "@rstest/core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook } from "@testing-library/react";
import { createElement, type ReactNode } from "react";

import { I18nContext } from "@/core/i18n/context";
import { enUS } from "@/core/i18n/locales/en-US";
import { isHiddenFromUIMessage } from "@/core/messages/utils";
import { DEFAULT_LOCAL_SETTINGS } from "@/core/settings/local";

/**
 * Combination verification for spec §4.4: incremental stream (messages-tuple /
 * updates / custom) against recovery boundaries — run creation, same-id
 * evolution, compaction REMOVE_ALL, journal-flush / history-refetch gaps,
 * replay-gap durable snapshots, and stop/error/finish final refreshes.
 *
 * The SDK is replaced by a faithful event-input fake: messages-tuple frames
 * perform same-id replacement and chunk assembly exactly like the SDK message
 * manager; updates frames flow through onUpdateEvent with the real mutate
 * contract. React itself is NOT mocked, and fake timers drive the ~80 ms
 * render-coalesce boundary so every frame is asserted deterministically.
 */

type UpdateMutate = (
  update:
    | Record<string, unknown>
    | ((previous: Record<string, unknown>) => Record<string, unknown>),
) => void;

type StreamOptions = {
  onCreated?: (meta: { thread_id: string; run_id: string }) => void;
  onUpdateEvent?: (data: unknown, options: { mutate: UpdateMutate }) => void;
  onCustomEvent?: (event: unknown) => void;
  onError?: (error: unknown) => void;
  onFinish?: (state: { values: { messages: Message[] } }) => void;
};

const mocks = rs.hoisted(() => ({
  fetch: rs.fn(),
  stream: {
    options: undefined as StreamOptions | undefined,
    isLoading: false,
    messages: [] as Message[],
    values: {
      artifacts: [] as string[],
      title: "",
      todos: [] as unknown[],
      goal: null as unknown,
    },
  },
}));

rs.mock("@langchain/langgraph-sdk/react", () => ({
  useStream: (options: StreamOptions) => {
    mocks.stream.options = options;
    return {
      isLoading: mocks.stream.isLoading,
      messages: mocks.stream.messages,
      stop: async () => undefined,
      submit: async () => undefined,
      values: { ...mocks.stream.values, messages: mocks.stream.messages },
    };
  },
}));

rs.mock("@/core/api/fetcher", () => ({ fetch: mocks.fetch }));

// ---------------------------------------------------------------------------
// SDK event-input emulation
// ---------------------------------------------------------------------------

function resetStream() {
  mocks.stream.options = undefined;
  mocks.stream.isLoading = false;
  mocks.stream.messages = [];
  mocks.stream.values = { artifacts: [], title: "", todos: [], goal: null };
}

/** messages-tuple frame: same-id replacement, otherwise append. */
function emitTuple(message: Message) {
  const index = mocks.stream.messages.findIndex((m) => m.id === message.id);
  if (index >= 0) {
    mocks.stream.messages = [
      ...mocks.stream.messages.slice(0, index),
      message,
      ...mocks.stream.messages.slice(index + 1),
    ];
  } else {
    mocks.stream.messages = [...mocks.stream.messages, message];
  }
}

/** messages-tuple chunk: the SDK assembles text deltas onto the same id. */
function emitChunk(id: string, delta: string) {
  const message = mocks.stream.messages.find((m) => m.id === id);
  if (!message) {
    throw new Error(`chunk for unknown message ${id}`);
  }
  if (typeof message.content !== "string") {
    throw new Error(`chunk for non-text message ${id}`);
  }
  emitTuple({ ...message, content: message.content + delta });
}

/** updates frame with the real mutate contract (values only, never messages). */
function emitUpdates(data: unknown) {
  mocks.stream.options?.onUpdateEvent?.(data, {
    mutate(update) {
      const patch =
        typeof update === "function" ? update(mocks.stream.values) : update;
      mocks.stream.values = { ...mocks.stream.values, ...patch };
    },
  });
}

/**
 * Compaction frame: SummarizationMiddleware emits RemoveMessage(ALL) + the
 * retained window; the SDK applies the removal to its message manager.
 */
function emitCompaction(retainedWindow: Message[]) {
  emitUpdates({
    "SummarizationMiddleware.before_model": {
      messages: [
        { type: "remove", id: "__remove_all__", content: "" } as Message,
        ...retainedWindow,
      ],
    },
  });
  mocks.stream.messages = [...retainedWindow];
}

function emitCustom(event: unknown) {
  mocks.stream.options?.onCustomEvent?.(event);
}

function emitFinish() {
  mocks.stream.options?.onFinish?.({
    values: { messages: mocks.stream.messages },
  });
  mocks.stream.isLoading = false;
}

function emitError(error: unknown) {
  mocks.stream.options?.onError?.(error);
  mocks.stream.isLoading = false;
}

function emitCreated(threadId: string, runId: string) {
  mocks.stream.options?.onCreated?.({ thread_id: threadId, run_id: runId });
}

// ---------------------------------------------------------------------------
// History feed emulation
// ---------------------------------------------------------------------------

type HistoryRow = { seq: number; run_id: string; content: Message };

function pageResponse(rows: HistoryRow[]) {
  return new Response(
    JSON.stringify({ data: rows, has_more: false, next_before_seq: null }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

let historyRows: HistoryRow[] = [];
let deferredHistory: ((response: Response) => void) | null = null;
let deferredHistoryPending = false;

function row(seq: number, runId: string, content: Message): HistoryRow {
  return { seq, run_id: runId, content };
}

/** Resolve a pending history refetch as the journal flush completing. */
function flushJournal(rows: HistoryRow[]) {
  historyRows = rows;
  deferredHistory?.(pageResponse(rows));
  deferredHistory = null;
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

function human(text: string, id: string): Message {
  return { type: "human", id, content: text } as Message;
}

function ai(text: string, id: string, extra?: Partial<Message>): Message {
  return { type: "ai", id, content: text, ...extra } as Message;
}

function withSeq(message: Message, seq: number): Message {
  return {
    ...message,
    additional_kwargs: { ...message.additional_kwargs, deerflow_seq: seq },
  } as Message;
}

async function setup(options: { isMock: boolean; threadId?: string }) {
  // Dynamic import: rs.mock above must be registered before the hooks module
  // (and its SDK import) is first evaluated.
  const { useThreadStream } = await import("@/core/threads/hooks");
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) =>
    createElement(
      QueryClientProvider,
      { client: queryClient },
      createElement(
        I18nContext.Provider,
        {
          value: {
            locale: "en-US",
            setLocale: () => undefined,
            t: enUS,
          },
        },
        children,
      ),
    );
  const hook = renderHook(
    () =>
      useThreadStream({
        context: DEFAULT_LOCAL_SETTINGS.context,
        isMock: options.isMock,
        threadId: options.threadId ?? "thread-1",
      }),
    { wrapper },
  );
  // Let the initial history page resolve.
  await flushActs();
  return hook;
}

type Hook = Awaited<ReturnType<typeof setup>>;
/** Drain react-query refetch/render cycles (microtask-driven, no timers). */
async function flushActs(times = 4) {
  for (let index = 0; index < times; index++) {
    await act(async () => {
      await rs.advanceTimersByTimeAsync(10);
    });
  }
}

/** Apply an SDK event, re-render, and advance past the coalesce interval. */
async function frame(hook: Hook, apply: () => void) {
  act(() => {
    apply();
    hook.rerender();
  });
  await act(async () => {
    void rs.advanceTimersByTime(100);
  });
}

function visibleMessages(hook: Hook): Message[] {
  return hook.result.current.thread.messages.filter(
    (message) => !isHiddenFromUIMessage(message),
  );
}

function frameIds(hook: Hook): Array<string | undefined> {
  return visibleMessages(hook).map((message) => message.id);
}

function expectUniqueIds(hook: Hook) {
  const ids = frameIds(hook).filter((id): id is string => Boolean(id));
  expect(new Set(ids).size).toBe(ids.length);
}

beforeEach(() => {
  // Only fake the clock and timeout APIs the coalesce/finalization logic
  // uses; faking setImmediate/MessageChannel would starve React's scheduler
  // and hang async act().
  rs.useFakeTimers({
    toFake: [
      "setTimeout",
      "clearTimeout",
      "setInterval",
      "clearInterval",
      "Date",
      "performance",
    ],
  });
  resetStream();
  historyRows = [];
  deferredHistory = null;
  deferredHistoryPending = false;
  mocks.fetch.mockImplementation(async (url: string | URL) => {
    if (String(url).includes("/messages/page")) {
      if (deferredHistoryPending) {
        return new Promise<Response>((resolve) => {
          deferredHistory = resolve;
        });
      }
      return pageResponse(historyRows);
    }
    return new Response("not found", { status: 404 });
  });
});

afterEach(() => {
  cleanup();
  rs.clearAllMocks();
  rs.useRealTimers();
});

// ---------------------------------------------------------------------------
// §4.4 — run creation and human confirmation
// ---------------------------------------------------------------------------

test("run creation: optimistic human stays ahead of early steps and is confirmed exactly once", async () => {
  const hook = await setup({ isMock: true });

  await act(async () => {
    await hook.result.current.sendMessage("thread-1", {
      files: [],
      text: "Build a presentation",
    });
  });
  // Frame 1: only the optimistic human is visible.
  expect(frameIds(hook)).toEqual([expect.stringMatching(/^opt-human-/)]);
  expectUniqueIds(hook);

  const earlyStep = ai("Reading the presentation skill", "run1-a1");
  await frame(hook, () => {
    emitCreated("thread-1", "run-1");
    mocks.stream.isLoading = true;
    emitTuple(earlyStep);
  });
  // Frame 2: the AI step arrived before the server human; the pending step
  // must stay behind this turn's (still optimistic) human.
  expect(frameIds(hook)).toEqual([
    expect.stringMatching(/^opt-human-/),
    "run1-a1",
  ]);

  // updates.messages carrying the same message must not be consumed a second
  // time (the SDK messages-tuple manager owns message state).
  await frame(hook, () => {
    emitUpdates({ agent: { title: "Presentation", messages: [earlyStep] } });
  });
  expect(frameIds(hook)).toEqual([
    expect.stringMatching(/^opt-human-/),
    "run1-a1",
  ]);
  expect(hook.result.current.thread.values.title).toBe("Presentation");

  // Frame 3: the server human copy arrives; the optimistic copy is retired.
  await frame(hook, () => {
    emitTuple(human("Build a presentation", "req-1__user"));
  });
  expect(frameIds(hook)).toEqual(["req-1__user", "run1-a1"]);
  const humans = visibleMessages(hook).filter((m) => m.type === "human");
  expect(humans).toHaveLength(1);
  expectUniqueIds(hook);
});

test("run creation with history beyond the checkpoint never reorders established history", async () => {
  historyRows = [
    row(1, "run-0", human("Earlier question", "h-old")),
    row(2, "run-0", ai("Earlier answer", "a-old")),
  ];
  const hook = await setup({ isMock: false });
  expect(frameIds(hook)).toEqual(["h-old", "a-old"]);

  await act(async () => {
    await hook.result.current.sendMessage("thread-1", {
      files: [],
      text: "Follow-up question",
    });
  });

  // The new turn's AI step arrives before the server human copy. The
  // established history must stay untouched and the pending step must wait
  // behind this turn's human.
  await frame(hook, () => {
    emitCreated("thread-1", "run-1");
    mocks.stream.isLoading = true;
    emitTuple(ai("Drafting the follow-up", "run1-a1"));
  });
  expect(frameIds(hook)).toEqual([
    "h-old",
    "a-old",
    expect.stringMatching(/^opt-human-/),
    "run1-a1",
  ]);

  await frame(hook, () => {
    emitTuple(human("Follow-up question", "req-2__user"));
  });
  expect(frameIds(hook)).toEqual(["h-old", "a-old", "req-2__user", "run1-a1"]);
  expectUniqueIds(hook);
});

// ---------------------------------------------------------------------------
// §4.4 — same id evolves from text into tool_calls
// ---------------------------------------------------------------------------

test("same id text -> tool_calls: single identity, stable position, content updates", async () => {
  historyRows = [
    row(1, "run-0", human("Question", "h-1")),
    row(2, "run-0", ai("Answer draft", "a-1")),
  ];
  const hook = await setup({ isMock: false });
  expect(frameIds(hook)).toEqual(["h-1", "a-1"]);

  // Durable snapshot after reconnect: same identities, no seq on live copies.
  await frame(hook, () => {
    mocks.stream.isLoading = true;
    emitTuple(human("Question", "h-1"));
    emitTuple(ai("Answer draft", "a-1"));
  });
  expect(frameIds(hook)).toEqual(["h-1", "a-1"]);

  // Chunk assembly extends the same id in place.
  await frame(hook, () => emitChunk("a-1", " continued"));
  expect(frameIds(hook)).toEqual(["h-1", "a-1"]);
  expect(visibleMessages(hook).find((m) => m.id === "a-1")?.content).toBe(
    "Answer draft continued",
  );

  // The same id grows tool_calls; still one message at the same position.
  await frame(hook, () =>
    emitTuple({
      type: "ai",
      id: "a-1",
      content: "",
      tool_calls: [{ id: "tc-1", name: "search", args: { q: "x" } }],
    } as Message),
  );
  expect(frameIds(hook)).toEqual(["h-1", "a-1"]);
  expectUniqueIds(hook);

  // The tool result associates by tool_call_id and lands after its call.
  await frame(hook, () =>
    emitTuple({
      type: "tool",
      id: "t-1",
      tool_call_id: "tc-1",
      content: "search result",
    } as Message),
  );
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "t-1"]);
  expectUniqueIds(hook);
});

test("live same-id copy without seq keeps the trusted history position metadata", async () => {
  historyRows = [
    row(1, "run-0", human("Question", "h-1")),
    row(2, "run-0", ai("Answer", "a-1")),
  ];
  const hook = await setup({ isMock: false });

  await frame(hook, () => {
    mocks.stream.isLoading = true;
    emitTuple(human("Question", "h-1"));
    // Live checkpoint copy carries no deerflow_seq; the trusted position from
    // the canonical feed must survive the content replacement.
    emitTuple(ai("Answer updated", "a-1"));
  });

  expect(frameIds(hook)).toEqual(["h-1", "a-1"]);
  const merged = visibleMessages(hook).find((m) => m.id === "a-1");
  expect(merged?.additional_kwargs?.deerflow_seq).toBe(2);
});

// ---------------------------------------------------------------------------
// §4.4 — compaction REMOVE_ALL / retained window
// ---------------------------------------------------------------------------

type CompactionScenario = {
  hook: Hook;
  historyAfterJournal: HistoryRow[];
};

async function runCompactionScenario(): Promise<CompactionScenario> {
  historyRows = [
    row(1, "run-0", human("First question", "h-1")),
    row(2, "run-0", ai("First answer", "a-1")),
  ];
  const hook = await setup({ isMock: false });

  // Checkpoint copies of the established history are already loaded, so the
  // local-turn baseline covers both identities.
  await frame(hook, () => {
    emitTuple(withSeq(human("First question", "h-1"), 1));
    emitTuple(withSeq(ai("First answer", "a-1"), 2));
  });

  await act(async () => {
    await hook.result.current.sendMessage("thread-1", {
      files: [],
      text: "Second question",
    });
  });

  await frame(hook, () => {
    emitCreated("thread-1", "run-1");
    mocks.stream.isLoading = true;
    emitTuple(withSeq(human("Second question", "h-2"), 3));
    emitTuple(withSeq(ai("Retained step", "a-2"), 4));
    emitTuple(withSeq(ai("Soon-summarized step", "a-3"), 5));
  });
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2", "a-3"]);

  // Compaction: RemoveMessage(ALL) + retained window [h-2, a-2]; a-3 is
  // dropped from the live checkpoint but was already rendered.
  await frame(hook, () => {
    emitCompaction([
      withSeq(human("Second question", "h-2"), 3),
      withSeq(ai("Retained step", "a-2"), 4),
    ]);
  });

  return {
    hook,
    historyAfterJournal: [
      row(1, "run-0", human("First question", "h-1")),
      row(2, "run-0", ai("First answer", "a-1")),
      row(3, "run-1", human("Second question", "h-2")),
      row(4, "run-1", ai("Retained step", "a-2")),
      row(5, "run-1", ai("Soon-summarized step", "a-3")),
    ],
  };
}

test("compaction REMOVE_ALL keeps the rescued turn visible across the persistence gap", async () => {
  const { hook } = await runCompactionScenario();

  // The summarized step stays visible (transient bridge) together with the
  // retained window and the established history — nothing the user saw
  // disappears while canonical history has not caught up.
  expect(frameIds(hook)).toContain("a-3");
  expect(frameIds(hook)).toContain("h-2");
  expect(frameIds(hook)).toContain("a-2");
  expect(frameIds(hook).slice(0, 2)).toEqual(["h-1", "a-1"]);
  expectUniqueIds(hook);
});

test("compaction REMOVE_ALL preserves the rescued turn's position after the retained window", async () => {
  const { hook } = await runCompactionScenario();

  // a-3 came after a-2 in the pre-compaction checkpoint (and both carry
  // trusted seqs), so the bridged frame must keep that order.
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2", "a-3"]);
});

test("compaction drains to single canonical copies once history catches up", async () => {
  const { hook, historyAfterJournal } = await runCompactionScenario();

  historyRows = historyAfterJournal;
  await frame(hook, () => emitFinish());
  await flushActs();

  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2", "a-3"]);
  expectUniqueIds(hook);
});

// ---------------------------------------------------------------------------
// §4.4 — journal not flushed / history refetch pending
// ---------------------------------------------------------------------------

test("finish with history refetch still pending keeps the bridged turn visible, then converges", async () => {
  const { hook, historyAfterJournal } = await runCompactionScenario();

  // The run finishes but the journal has not flushed: the history refetch
  // stays in flight. The bridged step must not disappear in the meantime.
  deferredHistoryPending = true;
  await frame(hook, () => emitFinish());
  await act(async () => {
    void rs.advanceTimersByTime(1000);
  });
  expect(frameIds(hook)).toContain("a-3");
  expect(frameIds(hook)).toContain("h-2");
  expectUniqueIds(hook);

  // The journal flush completes and the pending refetch resolves: canonical
  // history takes over with a single copy of every identity.
  deferredHistoryPending = false;
  await act(async () => {
    flushJournal(historyAfterJournal);
  });
  await flushActs();
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2", "a-3"]);
  expectUniqueIds(hook);
});

// ---------------------------------------------------------------------------
// §4.4 — replay gap: durable snapshot + incremental replay
// ---------------------------------------------------------------------------

test("replay gap: durable snapshot plus incremental replay keeps order and survives the gap event", async () => {
  historyRows = [
    row(1, "run-0", human("First question", "h-1")),
    row(2, "run-0", ai("First answer", "a-1")),
  ];
  const hook = await setup({ isMock: false });
  expect(frameIds(hook)).toEqual(["h-1", "a-1"]);

  // Reconnect: the durable snapshot already contains the active turn's human.
  await frame(hook, () => {
    mocks.stream.isLoading = true;
    emitTuple(withSeq(human("First question", "h-1"), 1));
    emitTuple(withSeq(ai("First answer", "a-1"), 2));
    emitTuple(withSeq(human("Second question", "h-2"), 3));
  });
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2"]);

  // Incremental replay continues with chunk assembly on the new steps.
  await frame(hook, () => emitTuple(ai("Step one", "a-2")));
  await frame(hook, () => emitChunk("a-2", " continued"));
  await frame(hook, () => emitTuple(ai("Step two", "a-3")));
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2", "a-3"]);
  expect(visibleMessages(hook).find((m) => m.id === "a-2")?.content).toBe(
    "Step one continued",
  );
  expectUniqueIds(hook);

  // The replay-gap boundary event clears transient state and forces a history
  // refresh; the durable snapshot plus history must keep the same visible
  // order and identities.
  const fetchesBefore = mocks.fetch.mock.calls.length;
  await frame(hook, () => emitCustom({ type: "stream_replay_gap" }));
  await flushActs();
  expect(mocks.fetch.mock.calls.length).toBeGreaterThan(fetchesBefore);
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2", "a-3"]);
  expectUniqueIds(hook);
});

// ---------------------------------------------------------------------------
// §4.4 — stop / error / finish final history refresh
// ---------------------------------------------------------------------------

async function runActiveTurn(): Promise<Hook> {
  historyRows = [
    row(1, "run-0", human("First question", "h-1")),
    row(2, "run-0", ai("First answer", "a-1")),
  ];
  const hook = await setup({ isMock: false });
  await frame(hook, () => {
    emitTuple(withSeq(human("First question", "h-1"), 1));
    emitTuple(withSeq(ai("First answer", "a-1"), 2));
  });
  await act(async () => {
    await hook.result.current.sendMessage("thread-1", {
      files: [],
      text: "Second question",
    });
  });
  await frame(hook, () => {
    emitCreated("thread-1", "run-1");
    mocks.stream.isLoading = true;
    emitTuple(withSeq(human("Second question", "h-2"), 3));
    emitTuple(withSeq(ai("Working on it", "a-2"), 4));
  });
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2"]);
  return hook;
}

test("finish triggers a final history refresh that converges to canonical order", async () => {
  const hook = await runActiveTurn();

  historyRows = [
    ...historyRows,
    row(3, "run-1", human("Second question", "h-2")),
    row(4, "run-1", ai("Working on it", "a-2")),
    row(5, "run-1", ai("Final answer", "a-3")),
  ];
  const fetchesBefore = mocks.fetch.mock.calls.length;
  await frame(hook, () => {
    emitTuple(withSeq(ai("Final answer", "a-3"), 5));
    emitFinish();
  });
  await flushActs();

  expect(mocks.fetch.mock.calls.length).toBeGreaterThan(fetchesBefore);
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2", "a-3"]);
  expectUniqueIds(hook);
});

test("stop triggers an immediate and a delayed final history refresh without reordering", async () => {
  // Dynamic import: see setup() — module must load after rs.mock registration.
  const { STOP_THREAD_FINALIZATION_REFETCH_DELAY_MS } =
    await import("@/core/threads/hooks");
  const hook = await runActiveTurn();

  historyRows = [
    ...historyRows,
    row(3, "run-1", human("Second question", "h-2")),
    row(4, "run-1", ai("Working on it", "a-2")),
  ];
  const fetchesBefore = mocks.fetch.mock.calls.length;
  await act(async () => {
    mocks.stream.isLoading = false;
    await hook.result.current.thread.stop();
  });
  await flushActs();
  const fetchesAfterStop = mocks.fetch.mock.calls.length;
  expect(fetchesAfterStop).toBeGreaterThan(fetchesBefore);
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2"]);

  // The scheduled finalization refetch catches a journal that flushed late.
  await act(async () => {
    void rs.advanceTimersByTime(STOP_THREAD_FINALIZATION_REFETCH_DELAY_MS);
  });
  await flushActs();
  expect(mocks.fetch.mock.calls.length).toBeGreaterThan(fetchesAfterStop);
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2"]);
  expectUniqueIds(hook);
});

test("error clears optimistic state and refreshes history without dropping visible turns", async () => {
  const hook = await runActiveTurn();

  historyRows = [
    ...historyRows,
    row(3, "run-1", human("Second question", "h-2")),
    row(4, "run-1", ai("Working on it", "a-2")),
  ];
  const fetchesBefore = mocks.fetch.mock.calls.length;
  await frame(hook, () => emitError(new Error("stream failed")));
  await flushActs();

  expect(mocks.fetch.mock.calls.length).toBeGreaterThan(fetchesBefore);
  // No optimistic leftovers, no lost visible messages.
  expect(frameIds(hook).some((id) => id?.startsWith("opt-"))).toBe(false);
  expect(frameIds(hook)).toEqual(["h-1", "a-1", "h-2", "a-2"]);
  expectUniqueIds(hook);
});

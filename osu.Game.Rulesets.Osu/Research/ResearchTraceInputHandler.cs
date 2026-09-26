using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HumanSim.Transport;
using osu.Framework.Input.StateChanges;
using osu.Framework.Platform;
using osu.Game.Replays;
using osu.Game.Rulesets.Osu.Replays;
using osu.Game.Rulesets.Replays;
using osuTK;

namespace osu.Game.Rulesets.Osu.Research
{
    /// <summary>Clock-driven, private synthetic input for X11 and Wayland research clients.</summary>
    public sealed class ResearchTraceInputHandler : OsuFramedReplayInputHandler
    {
        public int ConsumedFrames { get; private set; }
        public int ExpectedFrames { get; }
        public string? Failure { get; private set; }
        public bool Completed => ConsumedFrames == ExpectedFrames;
        private GameHost? host;
        private OsuReplayFrame? lastFrame;
        private readonly double settledTime;
        private double lastConsumedTime = double.NegativeInfinity;
        private Vector2? settledOrigin;
        private Vector2 settledXAxis;
        private Vector2 settledYAxis;

        public ResearchTraceInputHandler(ResearchTraceData trace)
            : base(createTimeline(trace))
        {
            FrameAccuratePlayback = true;
            ExpectedFrames = trace.Frames.Count;
            settledTime = (trace.TimelineStartEffectiveMs + 500) * trace.ClockRate;
        }

        public override bool Initialize(GameHost gameHost)
        {
            host = gameHost;
            return base.Initialize(gameHost);
        }

        protected override bool IsImportant(OsuReplayFrame frame) => true;

        protected override void CollectReplayInputs(List<IInput> inputs)
        {
            if (CurrentTime < lastConsumedTime) Failure ??= "Research gameplay clock moved backwards.";
            if (host != null && !host.IsActive.Value) Failure ??= "Research client lost focus.";
            if (CurrentTime >= settledTime && GamefieldToScreenSpace != null)
            {
                Vector2 origin = GamefieldToScreenSpace(Vector2.Zero);
                Vector2 xAxis = GamefieldToScreenSpace(new Vector2(512, 0));
                Vector2 yAxis = GamefieldToScreenSpace(new Vector2(0, 384));
                if (settledOrigin == null)
                {
                    settledOrigin = origin;
                    settledXAxis = xAxis;
                    settledYAxis = yAxis;
                }
                else if ((origin - settledOrigin.Value).Length > 2 || (xAxis - settledXAxis).Length > 2 || (yAxis - settledYAxis).Length > 2)
                    Failure ??= "Research playfield size or scale changed.";
            }
            if (Failure != null)
            {
                inputs.Add(new ReplayState<OsuAction> { PressedActions = new List<OsuAction>() });
                return;
            }
            base.CollectReplayInputs(inputs);
            if (CurrentFrame != null && CurrentFrame != lastFrame)
            {
                ConsumedFrames++;
                lastFrame = CurrentFrame;
                lastConsumedTime = CurrentTime;
            }
        }

        /// <summary>Exercise the actual handler with stalled render clocks, exact key edges and clock-rate transforms.</summary>
        public static string VerifyContracts()
        {
            int transportChecks = ResearchTraceData.VerifyContracts();
            int checks = 0;
            foreach (double rate in new[] { 0.75, 1.0, 1.5 })
            {
                var trace = new ResearchTraceData
                {
                    RunToken = "test", BeatmapSha256 = new string('a', 64), ClockRate = rate,
                    TimelineStartEffectiveMs = -500,
                    Frames = Enumerable.Range(0, 51).Select(i => new ResearchTraceFrame
                    { TimeUs = i * 1000, X = 100 + i, Y = 200, K1 = i is >= 10 and < 20, K2 = i is >= 25 and < 30 }).ToList(),
                };
                trace.Validate("test", new string('a', 64), rate);
                var handler = new ResearchTraceInputHandler(trace) { GamefieldToScreenSpace = p => p };
                var inputs = new List<IInput>();
                for (int i = 0; i < trace.Frames.Count; i++)
                {
                    // Simulate a render frame arriving after all recorded input.
                    double? time = handler.SetFrameFromTime(1000);
                    if (time != (-500 + i) * rate) throw new InvalidDataException("Synthetic input skipped a scheduled frame.");
                    inputs.Clear();
                    handler.CollectPendingInputs(inputs);
                    var position = inputs.OfType<MousePositionAbsoluteInput>().Single().Position;
                    var actions = inputs.OfType<ReplayState<OsuAction>>().Single().PressedActions;
                    if (position != new Vector2(100 + i, 200)
                        || actions.Contains(OsuAction.LeftButton) != trace.Frames[i].K1
                        || actions.Contains(OsuAction.RightButton) != trace.Frames[i].K2)
                        throw new InvalidDataException("Synthetic delivered cursor/key mismatch.");
                    checks++;
                }
                if (!handler.Completed || handler.ConsumedFrames != 51) throw new InvalidDataException("Synthetic frame accounting mismatch.");
                // Repeated polls must not inflate delivery exposure.
                handler.CollectPendingInputs(new List<IInput>());
                if (handler.ConsumedFrames != 51) throw new InvalidDataException("Duplicate synthetic frame counted.");
                handler.SetFrameFromTime(-10000);
                inputs.Clear();
                handler.CollectPendingInputs(inputs);
                if (handler.Failure == null || inputs.OfType<ReplayState<OsuAction>>().Single().PressedActions.Count != 0)
                    throw new InvalidDataException("Backward clock failed to release synthetic keys.");
            }
            return $"research_trace_checks={checks};transport_checks={transportChecks};rates=0.75,1,1.5;skipped_frames=0;key_edges=preserved";
        }

        private static Replay createTimeline(ResearchTraceData trace)
        {
            var timeline = new Replay();
            foreach (ResearchTraceFrame frame in trace.Frames)
            {
                var actions = new List<OsuAction>(2);
                if (frame.K1) actions.Add(OsuAction.LeftButton);
                if (frame.K2) actions.Add(OsuAction.RightButton);
                timeline.Frames.Add(new OsuReplayFrame((trace.TimelineStartEffectiveMs + frame.TimeUs / 1000.0) * trace.ClockRate,
                    new Vector2((float)frame.X, (float)frame.Y), actions.ToArray()));
            }
            return timeline;
        }
    }
}

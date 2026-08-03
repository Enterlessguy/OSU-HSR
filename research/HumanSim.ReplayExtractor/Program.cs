using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using osu.Game.Beatmaps;
using osu.Game.Rulesets;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Osu.Replays;
using osu.Game.Scoring.Legacy;

namespace HumanSim.ReplayExtractor;

public static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            Options options = Options.Parse(args);
            using FileStream replayStream = File.OpenRead(options.Replay);
            var decoder = new Decoder(options.Map);
            var score = decoder.Parse(replayStream);
            var frames = score.Replay.Frames.Cast<OsuReplayFrame>().ToArray();
            string salt = Environment.GetEnvironmentVariable("HUMAN_SIM_PLAYER_SALT")
                          ?? throw new InvalidOperationException("HUMAN_SIM_PLAYER_SALT must be set before extracting player data.");

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(options.Output))!);
            using var output = new StreamWriter(options.Output);
            output.WriteLine(JsonSerializer.Serialize(new
            {
                kind = "replay_header",
                schema_version = 1,
                beatmap_md5 = decoder.MapMd5,
                player_hash = hash(salt + "\0" + score.ScoreInfo.User.Username),
                mods = score.ScoreInfo.Mods.Select(m => m.Acronym).Order().ToArray(),
                frame_count = frames.Length,
            }));
            foreach (OsuReplayFrame frame in frames)
            {
                output.WriteLine(JsonSerializer.Serialize(new
                {
                    kind = "replay_frame",
                    time_ms = frame.Time,
                    x = frame.Position.X,
                    y = frame.Position.Y,
                    k1 = frame.Actions.Contains(OsuAction.LeftButton),
                    k2 = frame.Actions.Contains(OsuAction.RightButton),
                }));
            }
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception.Message);
            return 1;
        }
    }

    private static string hash(string value) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(value))).ToLowerInvariant();

    private sealed class Decoder : LegacyScoreDecoder
    {
        private readonly FlatWorkingBeatmap beatmap;
        public string MapMd5 { get; }

        public Decoder(string mapPath)
        {
            beatmap = new FlatWorkingBeatmap(mapPath);
            using FileStream stream = File.OpenRead(mapPath);
            MapMd5 = Convert.ToHexString(MD5.HashData(stream)).ToLowerInvariant();
        }

        protected override Ruleset GetRuleset(int rulesetId)
            => rulesetId == 0 ? new OsuRuleset() : throw new InvalidDataException("Only osu!standard replays are supported.");

        protected override WorkingBeatmap GetBeatmap(string md5Hash)
        {
            if (!StringComparer.OrdinalIgnoreCase.Equals(md5Hash, MapMd5))
                throw new InvalidDataException("Replay and beatmap MD5 hashes do not match.");
            return beatmap;
        }
    }

    private sealed record Options(string Replay, string Map, string Output)
    {
        public static Options Parse(string[] args)
        {
            var values = new Dictionary<string, string>();
            for (int i = 0; i < args.Length; i += 2)
            {
                if (i + 1 >= args.Length) throw new ArgumentException("Missing argument value.");
                values[args[i]] = args[i + 1];
            }
            return new Options(
                values.GetValueOrDefault("--replay") ?? throw new ArgumentException("--replay is required"),
                values.GetValueOrDefault("--map") ?? throw new ArgumentException("--map is required"),
                values.GetValueOrDefault("--output") ?? throw new ArgumentException("--output is required"));
        }
    }
}

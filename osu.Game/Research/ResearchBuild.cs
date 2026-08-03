// Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
// See the LICENCE file in the repository root for full licence text.

namespace osu.Game.Research
{
    using System.Collections.Generic;
    using System.Linq;
    using osu.Game.Rulesets.Mods;

    /// <summary>
    /// Compile-time marker for the isolated human-simulator client configuration.
    /// </summary>
    public static class ResearchBuild
    {
#if HUMAN_SIM_RESEARCH
        public static readonly bool Enabled = true;
#else
        public static readonly bool Enabled = false;
#endif

        public static bool AllowsLogin => !Enabled;

        public static bool AllowsScoreSubmission(IEnumerable<Mod> mods)
            => !Enabled && mods.All(mod => mod is not IResearchOnlyMod);
    }
}

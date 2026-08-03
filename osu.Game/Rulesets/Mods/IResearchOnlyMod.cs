// Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
// See the LICENCE file in the repository root for full licence text.

namespace osu.Game.Rulesets.Mods
{
    /// <summary>
    /// Marker for local research mods which must never retrieve a score token or submit a score.
    /// </summary>
    public interface IResearchOnlyMod : IApplicableMod
    {
    }

    /// <summary>
    /// Allows a research mod to synchronise an external process immediately before gameplay starts.
    /// </summary>
    public interface IResearchGameplayStartHook : IApplicableMod
    {
        /// <summary>
        /// Invoked immediately before the gameplay clock starts.
        /// </summary>
        /// <returns><see langword="true"/> when gameplay may start; otherwise <see langword="false"/>.</returns>
        bool OnGameplayStarting();
    }
}

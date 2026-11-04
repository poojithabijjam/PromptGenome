"""Strictly-improving prompt selection agent."""


class SelectionAgent:
    def run(self, incumbent, candidates):
        winner = max(candidates, key=lambda candidate: candidate.score, default=incumbent)
        return winner if winner.score > incumbent.score else incumbent

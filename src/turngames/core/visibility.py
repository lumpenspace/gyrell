from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VisibilityContext:
    """Who is asking to view a stateful object."""

    seat_id: str | None = None
    role: str | None = None
    team_id: str | None = None
    viewer: str = "player"


@dataclass(frozen=True)
class VisibilityRule:
    """Simple allow-list visibility rule.

    The rule is intentionally small. Games can layer richer policy on top, but
    the common case is "this board layer is visible to these roles/viewers".
    """

    always: bool = False
    roles: frozenset[str] = field(default_factory=frozenset)
    seat_ids: frozenset[str] = field(default_factory=frozenset)
    team_ids: frozenset[str] = field(default_factory=frozenset)
    viewers: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def public(cls) -> "VisibilityRule":
        return cls(always=True)

    @classmethod
    def for_roles(cls, *roles: str) -> "VisibilityRule":
        return cls(roles=frozenset(roles))

    @classmethod
    def for_viewers(cls, *viewers: str) -> "VisibilityRule":
        return cls(viewers=frozenset(viewers))

    def allows(self, context: VisibilityContext) -> bool:
        if self.always:
            return True
        if context.role and context.role in self.roles:
            return True
        if context.seat_id and context.seat_id in self.seat_ids:
            return True
        if context.team_id and context.team_id in self.team_ids:
            return True
        if context.viewer and context.viewer in self.viewers:
            return True
        return False

    def union(self, other: "VisibilityRule") -> "VisibilityRule":
        return VisibilityRule(
            always=self.always or other.always,
            roles=self.roles | other.roles,
            seat_ids=self.seat_ids | other.seat_ids,
            team_ids=self.team_ids | other.team_ids,
            viewers=self.viewers | other.viewers,
        )


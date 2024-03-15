from briefcase.enums import decision_enum
from briefcase.factor import Factor


class Case:
    """
    Class describing a case
    e.g. case 1
    pi_factors = frozenset{name="The child ate their dinner", polarity=pi}
    delta_factors = frozenset{name="The child didn't do their homework", polarity=delta}
    the decision = pi
    reason = name="The child ate their dinner", polarity=pi
    """
    @classmethod
    def from_reason_defeated(cls, new_reason, new_defeated):
        if list(new_reason)[0].polarity == decision_enum.pi:
            return cls(new_reason, new_defeated, decision_enum.pi, new_reason)
        else:
            return cls(new_defeated, new_reason, decision_enum.delta, new_reason)

    @classmethod
    def from_dict(cls, dic):
        """
        @param dic: a dictionary of 'pi', 'delta' and 'reason' factor lists, and a 'decision'
        @return: An instance of the Case class based on the dic.
        Converts pi, delta, and reason factor lists to frozensets (immutable sets) of tuples of each
        factor name and the polarity.
        """
        pi_factors = frozenset((Factor(f, decision_enum.pi) for f in dic["pi"]))
        delta_factors = frozenset(
            (Factor(f, decision_enum.delta) for f in dic["delta"])
        )

        # Check decision is valid
        try:
            decision_value = decision_enum[dic["decision"]]
        except KeyError as e:
            raise KeyError(
                f"The key 'decision' with value '{dic['decision']}' is not found in the decision_enum (pi/delta)."
            ) from e

        reason_factors = frozenset((Factor(f, decision_value) for f in dic["reason"]))

        # Check reason is valid
        winning_factors = (
            pi_factors if decision_value == decision_enum.pi else delta_factors
        )
        if not reason_factors.issubset(winning_factors):
            raise ValueError(
                f"reason_factors is not a subset of the winning_factors with the polarity {decision_value}"
            )

        return cls(pi_factors, delta_factors, decision_value, reason_factors)

    def __init__(
            self,
            pi_factors=frozenset(),
            delta_factors=frozenset(),
            decision=decision_enum.un,
            reason=frozenset(),
    ):
        """I think we can have as a degenerate case a pure decision, that is, one with no factors
        and thus no reasons. Alternatively, we could force all such to be undecided.

        If we allow nominal typing, then degenerate cases can be meaningfully different. We might use
        a general rule to express burden of proof, e.g., that we default for the defense.

        We might want to have an additional typology of cases to allow different burden of proofs for
        different sorts of cases, e.g., civil vs. criminal."""

        self.pi_factors = pi_factors
        self.delta_factors = delta_factors
        self.decision = decision
        self.reason = reason

    def defeated(self):  # TODO: make property
        """
        @return: the factors which were defeated by the decision e.g. for a decision of polarity pi, the delta factors
        """
        if self.decision == decision_enum.pi:
            return self.delta_factors
        elif self.decision == decision_enum.delta:
            return self.pi_factors

    def relevant_diff_from(self, other_case):
        """
        @param: other_case
        @return: Set of factors which are relevant differences from the other_case to this case
        This returns a set of relevant differences for factors of binary dimensions
        diff(other_case, this)
        """
        # outcome is the same
        if self.decision == other_case.decision:
            # get all reasons for this case which are in other case but not in this case
            reasons = other_case.reason - self.reason
            # get all defeated for this case which are in this case but not in other case
            defeated = self.defeated() - other_case.defeated()
        else:  # outcome is different
            # get all reasons for other case which are in other case but not in this one
            reasons = other_case.reason - self.defeated()
            # get all defeated in this case which are in other case but not in this case
            defeated = self.reason - other_case.defeated()
        return reasons | defeated

    def polar_opposite(self):
        """
        @return: Polar opposite of the current case, exactly the same factors but a decision in opposing direction
        """
        if self.decision == decision_enum.pi:
            return Case(self.pi_factors, self.delta_factors, decision_enum.delta, self.delta_factors)
        else:
            return Case(self.pi_factors, self.delta_factors, decision_enum.pi, self.pi_factors)

    def __str__(self):
        """
        @return: String representation of the Case for human-readable output
        """

        pi_factors_str = [str(factor) for factor in self.pi_factors]
        delta_factors_str = [str(factor) for factor in self.delta_factors]
        decision_str = f"Decision: {self.decision.name}"
        reason_str = [str(factor) for factor in self.reason]
        return (
            f"Case:\n  Pi Factors: {', '.join(pi_factors_str)}\n  Delta Factors: {', '.join(delta_factors_str)}\n  "
            f"{decision_str}\n  Reasons: {', '.join(reason_str)}"
        )

    def __repr__(self):
        """
        @return: String representation of the Case for recreation
        """
        pi_factors_repr = [repr(factor) for factor in self.pi_factors]
        delta_factors_repr = [repr(factor) for factor in self.delta_factors]
        decision_repr = (
            f"Decision.{self.decision.name}"  # Assuming 'decision' is an enum
        )
        reason_repr = [repr(factor) for factor in self.reason]
        return f"""Case(Pi Factors: {pi_factors_repr}, Delta Factors: {delta_factors_repr}, {
        decision_repr}, Reasons: {reason_repr})"""


    def __eq__(self, other):
        if isinstance(other, Case):
            return (
                self.pi_factors == other.pi_factors and
                self.delta_factors == other.delta_factors and
                self.decision == other.decision and
                self.reason == other.reason
            )
        return False

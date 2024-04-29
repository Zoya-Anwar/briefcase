from collections import defaultdict
from itertools import combinations
from briefcase.power_detector import PowerDetector
from briefcase.admissibility_constraints import AdmissibilityConstraints
from briefcase.enums import incons_enum, decision_enum


class PriorityOrder:
    """
    Orders reasons and factors in a dictionary.
    Key is a frozenset of the stronger factors, value is a frozenset of the weaker factors.
    """

    def __init__(self, cb, empty_sides=False):
        self.cb = cb
        self.order = defaultdict(set)
        self.defeated_factor_index = defaultdict(set)
        self.admissibility_constraints = AdmissibilityConstraints(self)
        self.PD = PowerDetector(self)
        self.empty_sides=empty_sides

    def get_cases(self):
        return self.cb.cases

    def get_incons_pairs_with_case(self, new_reason, new_defeated):
        """
        @param new_reason: a frozenset of factors
        @param new_defeated: a frozenset of factors, weaker than the reason
        @return: list of pairs of reasons, defeated which are inconsistent with a new case
        """
        # ... P > D (subsets)
        # we want to find all P2, D2 where P2 > P and D2 < D
        # D2 > P2
        # get all defeats which are stronger than the reason

        supersets = self.get_stronger_defeats(new_reason)
        case_sets = [(self.order[superset], superset) for superset in supersets]

        incons_pairs = []
        for case in case_sets:
            reasons = case[0]
            defeated = case[1]
            for reason in reasons:
                if reason.issubset(new_defeated):
                    incons_pairs.append((reason, defeated))

        return incons_pairs

    def is_existing_claim(self, new_reason, new_defeated):
        """Checks priority order remains the same"""
        # if there exists a case with a stronger than or equal to defeated
        for stronger_defeat in self.get_stronger_defeats(new_defeated):
            # and with a weaker or equal to reason
            if any(new_reason.issuperset(reason) for reason in self.order[stronger_defeat]):
                return True
        return False

    def get_stronger_defeats(self, factor_set):
        # Retrieve all entries in defeated_factor_index for all factors within factor_set (defeats_that_intersect)
        # Cast existing set of references to a frozenset within a set
        defeats_that_intersect = {frozenset(self.defeated_factor_index[factor]) for factor in factor_set}

        # Find the intersection of all sets retrieved above, exploiting the references property
        # This gives us all sets of factors which are stronger than our factor_set - supersets
        # To optimise this, we order by size so that initially smaller sets are compared for intersections
        sorted_defeats_that_intersect = sorted(defeats_that_intersect, key=lambda x: len(x))
        if sorted_defeats_that_intersect:
            supersets = frozenset.intersection(*sorted_defeats_that_intersect)
            return supersets
        return []

    # def get_weaker_defeats(self, factor_set):
    #     # Retrieve all entries in defeated_factor_index which are weaker than a given factor set
    #     # This any factor sets containing any number of the factors in the factor set
    #
    #     """
    #     How do I find all subsets of a factorset e.g. {d1, d2, d3},
    #     d1 ={{d1, d3}, {d1, d2}}
    #     d2 = {{d1, d2}, {d2, d5}}
    #     d3 = {{d1, d3}, {d4}}
    #     d4 = {{d4}}
    #
    #     I don't think you can efficiently. Let's just loop through
    #     """
    #
    #     # Initialize an empty set to store the union of all sets
    #     weaker_defeats = set()
    #
    #     # Iterate through each factor in factor_set and update the union
    #     for factor in factor_set:
    #         candidate_weaker_defeats = self.defeated_factor_index[factor]
    #         for candidate in candidate_weaker_defeats:
    #             if candidate.issubset(factor_set):
    #                 weaker_defeats.add(candidate)
    #
    #     return weaker_defeats
    def add_order_with_subsets(self, reason, defeated):
        """
        @param reason: a frozenset of factors
        @param defeated: a frozenset of factors, weaker than the reason
        Adds defeated: reason to the cb order.
        Adds all factors within the defeated set to the defeated_subsets dictionary with the
        defeated frozenset as key and a set of frozensets as values
        """

        if reason != frozenset() and defeated != frozenset():
            self.order[defeated].add(reason)

            for factor in defeated:
                self.defeated_factor_index[factor].add(defeated)

    def is_consistent(self, new_reason, new_defeated):
        """
        @param new_reason: a frozenset of factors
        @param new_defeated: a frozenset of factors, weaker than the reason
        @return: True/False if for the new_reason being stronger than the new_defeated,
                this causes inconsistency with the existing Case Base order
        """
        supersets = self.get_stronger_defeats(new_reason)

        # Looking at each (defeated) superset within the set of supersets,
        # Retrieve the (reason) entry in the existing order dictionary - this will be a set of reasons for each
        # Intersecting case
        reasons_sets = [self.order[superset] for superset in supersets]

        # If for any of the existing reasons sets retrieved, if the new defeated is a superset of one of
        # the sets of old reasons, return False (inconsistent)
        return not any(any(reason.issubset(new_defeated) for reason in reasons) for reasons in reasons_sets)

    def is_cb_consistent(self):
        """
        @return : True/False if current ordering of the cb order is consistent
        Inconsistency must be strict
        """
        # loop through all cases in order
        for defeated, reason_set in self.order.items():
            for reason in reason_set:
                if not self.is_consistent(reason, defeated):
                    return False
        return True

    def is_cb_consistent_with(self, case):
        """
        @param case: a new case
        @return: True/False if the current cb order would be consistent with a new case added
        """
        return self.is_consistent(case.reason, case.defeated())

    def unsafe_add_case(self, case):
        """
        @param case: the new case to be added to the priority order
        Adds a new case to order dict with no safety checks for inconsistency
        """
        # case 1: we know the winning reason is at least as strong as the defeated factors, since it won
        if (case.reason and case.defeated()) or self.empty_sides: # cannot have an empty side
            self.add_order_with_subsets(case.reason, case.defeated())
            self.PD.add_factor_list(case)
            return True
        return False

    def safe_add_case(self, case, incons):
        """
        @param case: the new case to be added to the priority order
        @param incons: the constraint to be used when adding the new case to the priority order
        """
        if self.admissibility_constraints.is_case_admissible(case.reason, case.defeated(), incons):
            self.unsafe_add_case(case)
            return True
        return False

    def str_order(self):
        """
        @return: String representation of the PriorityOrder for human-readable output
        """
        cases_str = []
        for winning, defeated in self.order.items():
            cases_str.append("Reason: " + str(winning) + ", Defeated: " + str(defeated))
        cases_formatted = "\n".join(cases_str)
        return f"\nPriority Order:\n{cases_formatted}"

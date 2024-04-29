from enum import Enum

decision_enum = Enum("Decision", ["pi", "delta", "un"])
incons_enum = Enum("Inconsistency", ["NO_INCONSISTENCY", "NO_CHANGED_ORDER", "NO_INVOLVEMENT",
                                                  "NO_NEW", "NO_CORRUPTION", "MRD", "INTERSECTING_EDGES", "ALL"])

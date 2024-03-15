from enum import Enum

decision_enum = Enum("Decision", ["pi", "delta", "un"])
incons_enum = Enum("Inconsistency", ["NO", "NO_NEW", "NO_INVOLVEMENT", "HORTY", "NO_CORRUPTION",
                                                  "MRD", "INTERSECTING_EDGES", "ALL"])

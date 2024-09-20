from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

def create_model(clause: int, T: int, sensitivty: float):
    model = TMCoalescedClassifier(clause, T, sensitivty, weighted_clauses=True)
    return model
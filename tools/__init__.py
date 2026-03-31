from .validate_solution import validate_payload_solution

def analyze_solution_payload(*args, **kwargs):
    from .analyze_solution import analyze_solution_payload as impl

    return impl(*args, **kwargs)


def analyze_solution_parts(*args, **kwargs):
    from .analyze_solution import analyze_solution_parts as impl

    return impl(*args, **kwargs)


__all__ = ["analyze_solution_payload", "analyze_solution_parts", "validate_payload_solution"]

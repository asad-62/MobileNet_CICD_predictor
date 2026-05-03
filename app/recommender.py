

SUNGLASSES_RECOMMENDATIONS = {
    "heart": ["Aviator", "Cat-eye"],
    "long": ["Oversized", "Square/Rectangular"],
    "oval": ["Square/Rectangular", "Cat-eye"],
    "round": ["Square/Rectangular", "Cat-eye"],
    "square": ["Round/Oval", "Aviator"],
}

def recommend_sunglasses(face_type: str) -> list[str]:
    """
    Return sunglasses recommendations for a predicted face type.
    """
    return SUNGLASSES_RECOMMENDATIONS.get(face_type, [])

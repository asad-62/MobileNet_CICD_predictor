from app.recommender import recommend_sunglasses


def test_recommend_sunglasses_for_oval():
    result = recommend_sunglasses("oval")

    assert result == ["Square/Rectangular", "Cat-eye"]


def test_recommend_sunglasses_for_unknown_face_type():
    result = recommend_sunglasses("unknown")

    assert result == []


    
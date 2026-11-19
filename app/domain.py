"""Cafeteria identities shared by collection and question filtering.

Only official facility identities are constants here. Opening hours and prices
can change and must be read from dated source documents rather than hardcoded.
"""

RESTAURANTS = {
    "re11": {"name": "교직원식당", "facility_id": "2", "aliases": ["교식", "교직원식당"]},
    # "학식" describes campus meals generally, not this particular cafeteria.
    "re12": {"name": "학생식당", "facility_id": "1", "aliases": ["학생식당"]},
    "re13": {
        "name": "창의인재원식당", "facility_id": "4",
        "aliases": ["창의", "창의인재", "긱식", "기숙사식당", "창의인재원식당"],
    },
    "re15": {
        "name": "창업보육센터식당", "facility_id": "3",
        "aliases": ["창보", "창업보육센터", "창업보육센터식당"],
    },
}

MEAL_TYPES = ("조식", "중식", "석식")


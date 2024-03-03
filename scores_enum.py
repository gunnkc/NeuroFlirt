from enum import Enum

# Overall 0-10
class Score(Enum):
    TERRIBLE = 0
    POOR = 1
    VERY_BAD = 2
    BAD = 3
    BELOW_AVERAGE = 4
    AVERAGE = 5
    ABOVE_AVERAGE = 6
    GOOD = 7
    VERY_GOOD = 8
    EXCELLENT = 9
    GREAT = 10
    

# Boring 0-10
class Engagement(Enum):
    UNENGAGED = 0
    BARELY_INTERESTED = 1
    SLIGHTLY_INTERESTED = 2
    SOMEWHAT_INTERESTED = 3
    MILDLY_INTERESTED = 4
    MODERATELY_INTERESTED = 5
    INTERESTED = 6
    HIGHLY_INTERESTED = 7
    VERY_INTERESTED = 8
    EXTREMELY_INTERESTED = 9
    VERY_INTERESTED = 10

# Arousal 0-8
class Attraction(Enum):
    UNATTRACTED = 0
    VERY_LOW = 1
    LOW = 2
    SOMEWHAT_LOW = 3
    NEUTRAL = 4
    SOMEWHAT_HIGH = 5
    HIGH = 6
    VERY_HIGH = 7
    ATTRACTED = 8

# Valence 0-8
class Satisfaction(Enum):
    SAD = 0
    UNHAPPY = 1
    SOMEWHAT_UNHAPPY = 2
    NEUTRAL = 3
    SLIGHTLY_CONTENT = 4
    CONTENT = 5
    HAPPYISH = 6
    VERY_CONTENT = 7
    HAPPY = 8
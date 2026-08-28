from dataclasses import dataclass


@dataclass
class Movie:
    title: str
    genre: str


@dataclass
class Recommendation:
    movie: Movie
    reason: str


def build_recommendation_summary(recommendations: list[Recommendation]) -> tuple[str, list[Movie]]:
    """Build a display summary and the list of movies it's built from.

    Args:
        recommendations: Recommendations to summarize, in order.

    Returns:
        (summary, movies_used) -- summary is the display string;
        movies_used is built directly from recommendations, not
        parsed back out of summary.
    """
    summaries = [
        f"{recommendation.movie.title} ({recommendation.movie.genre}): {recommendation.reason}"
        for recommendation in recommendations
    ]
    movies_used = [recommendation.movie for recommendation in recommendations]
    return "\n\n".join(summaries), movies_used



if __name__ == "__main__":
    movie_recommendations = [
        Recommendation(
            movie=Movie(title="Inception", genre="Sci-Fi"),
            reason="It has a brilliant plot that pushes the boundaries of the mind with its dream-within-a-dream concept."
        ),
        Recommendation(
            movie=Movie(title="The Shawshank Redemption", genre="Drama"),
            reason="Built on hope and friendship, it is one of the highest-rated masterpieces in cinema history."
        ),
        Recommendation(
            movie=Movie(title="The Dark Knight", genre="Action"),
            reason="The pinnacle of comic book adaptations, featuring Heath Ledger's legendary Joker performance and a deep underworld story."
        ),
        Recommendation(
            movie=Movie(title="Interstellar", genre="Sci-Fi"),
            reason="It explores space-time warping and black holes with magnificent visual effects and music."
        ),
        Recommendation(
            movie=Movie(title="Parasite", genre="Thriller"),
            reason="An Oscar-winning South Korean production that portrays class conflict through a humorous and suspenseful lens."
        ),
        Recommendation(
            movie=Movie(title="Spirited Away", genre="Animation"),
            reason="A Miyazaki classic with a captivating atmosphere that pushes the limits of imagination."
        ),
        Recommendation(
            movie=Movie(title="Whiplash", genre="Drama"),
            reason="It demonstrates what can be sacrificed for the sake of success and ambition with flawless rhythm and acting."
        ),
        Recommendation(
            movie=Movie(title="Se7en", genre="Mystery"),
            reason="A cult detective thriller based on the seven deadly sins that catches the audience off guard with its twist ending."
        ),
        Recommendation(
            movie=Movie(title="Gladiator", genre="Action"),
            reason="It presents a story of revenge during the Roman Empire with epic battle scenes."
        ),
        Recommendation(
            movie=Movie(title="Coco", genre="Animation"),
            reason="An emotional film that explores family bonds and the power of music through a colorful visual feast."
        )
    ]

    summary, movis_used = build_recommendation_summary(movie_recommendations)
    print(summary)
    print()
    print(*movis_used, sep="\n")


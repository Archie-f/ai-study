def make_discounter(rate: float):
    """Return a function apply(price) that takes rate off the price.

    Example: ten_off = make_discounter(0.10); ten_off(200.0) -> 180.0
    """
    def discounted_price(price: float) -> float:
        return price * (1 - rate)
    return discounted_price


if __name__ == "__main__":
    ten_off = make_discounter(0.10)
    print(ten_off(200.0))   # 180.0
    print(ten_off(50.0))    # 45.0

    none_off = make_discounter(0.0)
    print(none_off(80.0))   # 80.0

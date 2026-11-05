from verify.verify_agent_tool_use import tool_call_rates


def test_no_rows_gives_empty_dict():
    assert tool_call_rates([]) == {}


def test_example_from_the_docstring():
    rows = [("notes", True), ("notes", False), ("controls", False)]
    assert tool_call_rates(rows) == {"notes": (1, 2), "controls": (0, 1)}


def test_rows_of_one_group_do_not_have_to_be_next_to_each_other():
    rows = [("notes", True), ("controls", False), ("notes", True), ("controls", False)]
    assert tool_call_rates(rows) == {"notes": (2, 2), "controls": (0, 2)}


def test_all_called():
    assert tool_call_rates([("a", True), ("a", True), ("a", True)]) == {"a": (3, 3)}


def test_none_called_is_zero_not_missing():
    assert tool_call_rates([("a", False), ("a", False)]) == {"a": (0, 2)}


def test_groups_appear_in_the_order_first_seen():
    rows = [("controls", False), ("notes", True), ("controls", True)]
    assert list(tool_call_rates(rows)) == ["controls", "notes"]


def test_groups_are_counted_separately():
    rows = [("a", True), ("b", False), ("a", False), ("b", False), ("b", True)]
    assert tool_call_rates(rows) == {"a": (1, 2), "b": (1, 3)}


def test_group_name_that_is_a_prefix_of_another_is_a_separate_group():
    rows = [("notes", True), ("notes-extra", False)]
    assert tool_call_rates(rows) == {"notes": (1, 1), "notes-extra": (0, 1)}


def test_input_list_is_not_changed():
    rows = [("a", True), ("a", False)]
    copy = list(rows)
    tool_call_rates(rows)
    assert rows == copy


def test_result_values_are_ints():
    called, total = tool_call_rates([("a", True)])["a"]
    assert type(called) is int and type(total) is int

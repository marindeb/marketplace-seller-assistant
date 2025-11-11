from app import agent


def test_load_data_structure():
    """
    Basic unit test to ensure that the data loading function works properly.
    It checks that:
    - the returned DataFrame is not empty,
    - and that key columns ('category', 'return_rate') are present.
    """
    df = agent.load_data()

    # The DataFrame should contain data
    assert not df.empty, "DataFrame should not be empty."

    # Essential columns must exist
    expected_cols = {"product_id", "late_rate"}
    assert expected_cols.issubset(
        df.columns
    ), f"Missing expected columns: {expected_cols - set(df.columns)}"

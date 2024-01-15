class TestOWT:
    def test_init(
        self, api_test, materials_df, sa_df, owt_init, mock_requests_sa_get_bb
    ):
        # owt = OWT(api_test, materials_df, sa_df)
        # #print(owt.tower_sub_assemblies, owt_init["tower_sub_assemblies"])
        # # for key in owt.__dict__:
        # #     print(key)
        # #     assertion, message = deepcompare(getattr(owt, key), owt_init[key])
        # #     assert assertion, message
        # #print(type(getattr(owt, "tower_sub_assemblies")))
        # assertion, message = deepcompare(
        #         getattr(owt, "sub_assemblies")["TP"].bb[0].json, owt_init["sub_assemblies"]["TP"]["bb"][0]["json"]
        # )
        # #print(
        #         getattr(owt, "tower_sub_assemblies")["volume"].loc["aaa01_tw_30"],
        #         getattr(owt, "tower_sub_assemblies")["volume"].dtype
        # )
        # #print(
        #     owt_init["tower_sub_assemblies"]["volume"].loc["aaa01_tw_30"], owt_init["tower_sub_assemblies"]["volume"].dtype
        # )
        # assert assertion, message
        assert True

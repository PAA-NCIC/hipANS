// gtest_sum.cpp
#include <iostream>
#include <gtest/gtest.h>

int sum(int a, int b) {
	return a+b;
}

TEST(sum, testSum) {
	EXPECT_EQ(5, sum(2, 3));	// 求合2+3=5
	EXPECT_NE(3, sum(3, 4));	// 求合3+4 != 3
}
// 如果在此处不写main函数，那么在链接库的时候还需要链接-lgtest_main, 否则只需链接-lgtest即可。
#if 0
int main(int argc, char **argv)
{
 testing::InitGoogleTest(&argc, argv);
 return RUN_ALL_TESTS();
}
#endif


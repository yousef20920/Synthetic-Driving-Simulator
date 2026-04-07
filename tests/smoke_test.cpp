// Phase 1 scaffold smoke test: verifies the build and test harness work.
#include <gtest/gtest.h>

TEST(ScaffoldSmoke, BuildSystemWorks) {
    EXPECT_EQ(1 + 1, 2);
}

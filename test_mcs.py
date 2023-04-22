from bayes_rule_MCS import monte_carlo_twice, monte_carlo_once
import unittest


class TestMCS(unittest.TestCase):

    def test_monte_carlo_twice(self):
        self.assertEqual(monte_carlo_twice(0.1, 0.3, 0.6), 3)
        self.assertEqual(monte_carlo_twice(0.4, 0.3, 0.3), 1)
        self.assertEqual(monte_carlo_twice(0.1, 0.7, 0.2), 2)
        self.assertIn(monte_carlo_twice(0.2, 0.4, 0.4), [2, 3])
        self.assertIn(monte_carlo_twice(0.4, 0.4, 0.2), [1, 2])
        self.assertIn(monte_carlo_twice(0.4, 0.2, 0.4), [1, 3])

    def test_monte_carlo_once(self):
        self.assertEqual(monte_carlo_once(0.1, 0.3, 0.6), 6)
        self.assertEqual(monte_carlo_once(0.4, 0.3, 0.4), 5)
        self.assertEqual(monte_carlo_once(0.4, 0.4, 0.2), 4)
        self.assertIn(monte_carlo_once(0.33, 0.33, 0.33), [4, 5, 6])
        self.assertEqual(monte_carlo_once(0.4, 0.4, 0.2), 4)
        self.assertEqual(monte_carlo_once(0.4, 0.2, 0.4), 5)
        self.assertEqual(monte_carlo_once(0.2, 0.4, 0.4), 6)
        self.assertIn(monte_carlo_once(0.2, 0.2, 0.6), [5, 6])
        self.assertIn(monte_carlo_once(0.2, 0.6, 0.2), [4, 6])
        self.assertIn(monte_carlo_once(0.6, 0.2, 0.2), [4, 5])

if __name__ == '__main__':
    unittest.main()
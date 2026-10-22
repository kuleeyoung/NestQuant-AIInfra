import unittest

from nestquant.core.lattice import nearest_e8 as domain_nearest_e8
from nestquant.lattice import nearest_e8 as compatibility_nearest_e8
from nestquant import nearest_e8 as top_level_nearest_e8
from nestquant import PackedNestQuantModel as top_level_model_checkpoint
from nestquant.core.model_checkpoint import PackedNestQuantModel as domain_model_checkpoint


class TestPublicImports(unittest.TestCase):
    def test_old_and_top_level_imports_reexport_domain_implementation(self):
        self.assertIs(compatibility_nearest_e8, domain_nearest_e8)
        self.assertIs(top_level_nearest_e8, domain_nearest_e8)
        self.assertIs(top_level_model_checkpoint, domain_model_checkpoint)


if __name__ == "__main__":
    unittest.main()

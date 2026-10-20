import unittest
from prefix_selection import choose_base


class SelectionTests(unittest.TestCase):
    def test_two_profiles_updates_front_edits_and_exact_compatibility(self):
        a=list(range(1000,2000));b=list(range(3000,4000))
        entries=[{'id':'base-1','tokens':a,'runtime':'v1','valid':True,'last_used':20},
                 {'id':'base-2','tokens':b,'runtime':'v1','valid':True,'last_used':10}]
        for tokens,expected in [(a,'base-1'),(b,'base-2')]:
            self.assertEqual(choose_base(tokens,entries,runtime='v1')['id'],expected)
            self.assertEqual(choose_base(tokens,entries,runtime='v1')['mode'],'exact')
        # A tools change at the very front must not misclassify the unchanged core.
        changed=[99999]+a[1:]
        selected=choose_base(changed,entries,runtime='v1')
        self.assertEqual((selected['id'],selected['mode']),('base-1','replace'))
        entries[0]['valid']=False
        self.assertEqual(choose_base(a,entries,runtime='v1')['mode'],'replace')
        self.assertEqual(choose_base(b,entries,runtime='v2')['mode'],'replace')
        self.assertEqual(choose_base(b,entries[:1],runtime='v1')['id'],'base-2')

    def test_true_tie_is_explicit_and_source_identity_can_disambiguate(self):
        entries=[{'id':'base-1','tokens':list(range(100,200)),'last_used':20,'source_identity':'web'},
                 {'id':'base-2','tokens':list(range(300,400)),'last_used':10,'source_identity':'telegram'}]
        unrelated=list(range(1000,1100))
        selected=choose_base(unrelated,entries)
        self.assertTrue(selected['ambiguous'])
        self.assertEqual(selected['id'],'base-2')
        selected=choose_base(unrelated,entries,source_identity='web')
        self.assertFalse(selected['ambiguous'])
        self.assertEqual(selected['id'],'base-1')


if __name__=='__main__':unittest.main()

import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest


class SlicerMuscleContoursTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.setUp()
        self.test_CreateContourMetadata()

    def setUp(self):
        slicer.mrmlScene.Clear()

    def test_CreateContourMetadata(self):
        from SlicerMuscleContours import SlicerMuscleContoursLogic

        volume = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", "Reference"
        )
        logic = SlicerMuscleContoursLogic()
        node = logic.createContourNode(
            volume, "Muscle", "Red", 12, [0, 0, 12], [0, 0, 1]
        )
        self.assertEqual(node.GetAttribute(logic.ATTR_GROUP), "Muscle")
        self.assertEqual(node.GetAttribute(logic.ATTR_SLICE_INDEX), "12")
        self.assertEqual(node.GetNodeReferenceID(logic.ROLE_REFERENCE), volume.GetID())

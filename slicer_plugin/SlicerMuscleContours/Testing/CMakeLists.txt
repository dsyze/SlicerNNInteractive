if(NOT BUILD_TESTING)
  return()
endif()

slicer_add_python_unittest(
  SCRIPT Python/SlicerMuscleContoursTest.py
)

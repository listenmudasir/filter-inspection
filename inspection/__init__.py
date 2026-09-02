"""Runtime preprocessing shared by the GUI and the detector.

Only enhance.py is required by the supervised method: crop_to_content and
illumination_correct are the EXACT transform the training data was built with.
Substituting or skipping them feeds the model a distribution it never saw.

The old hybrid stack's modules (service, runtime, features, model, tiling,
compat) are deliberately NOT shipped. That method is superseded and carrying it
would mean shipping an 80 MB backbone nothing calls.
"""

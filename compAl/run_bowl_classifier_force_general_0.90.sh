
clear

cd bowl_classifier
python bowl_classifier_phase1_majorclass_inference.py force 0.9
python bowl_classifier_phase2_histology_inference.py force 0.9
cd ..


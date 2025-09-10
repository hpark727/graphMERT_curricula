#! /bin/bash
python generate_curriculum.py --max_k_hops 3 --num_questions 24000 --output_dir /curriculum_training_data/ --api_key $GEMINI_API_KEY

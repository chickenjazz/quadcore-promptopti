# Install PyTorch with CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Target Prompt Structure:

You are an experienced home baker and cooking instructor.
Explain how to bake classic chocolate brownies from scratch for a beginner.
Include:

A complete list of ingredients with exact measurements

Required kitchen tools

Step-by-step instructions in correct order

Oven temperature and baking time

Tips to achieve a fudgy texture

Common mistakes to avoid

- Has a role
- specific task
- examples and constraints
from pathlib import Path

def test_dataset_existence():
  assert Path("model/wine_quality.csv").is_file(), "There is no dataset 'wine_quality.csv' to apply model on."

def test_a_and_b():    
  assert 5+5==10, "Math is broken!"
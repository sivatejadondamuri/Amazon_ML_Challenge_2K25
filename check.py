import pandas as pd

test = pd.read_csv(r"d:\student_resource\dataset\test.csv")
pred = pd.read_csv(r"d:\student_resource\dataset\test_out.csv")

print("Same length:", len(test) == len(pred))
print("Same order:", (test['sample_id'] == pred['sample_id']).all())


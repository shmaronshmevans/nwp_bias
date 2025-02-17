import pandas as pd

oksm = pd.read_csv("/home/aevans/nwp_bias/src/landtype/src/ok_mesonet.csv")

for c in oksm.columns:
    print(c)

print(oksm)

oksm.to_parquet("/home/aevans/nwp_bias/src/landtype/src/ok_mesonet.parquet")

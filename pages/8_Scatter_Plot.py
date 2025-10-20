# ============================================================
# 🔧 CLEAN COLUMN HEADERS & REMOVE IRRELEVANT METRICS
# ============================================================
df_all.columns = (
    df_all.columns.astype(str)
    .str.strip()
    .str.replace(u"\xa0", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

# Drop unwanted columns: any containing "Player Season" or "Account Id"
drop_pattern = re.compile(r"(?i)(player season|account id)")
cols_to_drop = [c for c in df_all.columns if drop_pattern.search(c)]
if cols_to_drop:
    print(f"[DEBUG] Dropping {len(cols_to_drop)} columns: {cols_to_drop[:5]} ...")
    df_all.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# ============================================================
# 🧩 MAP TO SIX GROUP POSITIONS
# ============================================================
def _clean_pos_token(tok: str) -> str:
    if pd.isna(tok):
        return ""
    t = str(tok).upper().strip()
    t = re.sub(r"[.\-_/]", " ", t)
    t = re.sub(r"\s+", "", t)
    return t

RAW_TO_SIX = {
    "RIGHTBACK": "Full Back", "LEFTBACK": "Full Back",
    "RIGHTWINGBACK": "Full Back", "LEFTWINGBACK": "Full Back",
    "RIGHTCENTREBACK": "Centre Back", "LEFTCENTREBACK": "Centre Back", "CENTREBACK": "Centre Back",
    "DEFENSIVEMIDFIELDER": "Number 6", "RIGHTDEFENSIVEMIDFIELDER": "Number 6",
    "LEFTDEFENSIVEMIDFIELDER": "Number 6", "CENTREDEFENSIVEMIDFIELDER": "Number 6",
    "CENTREMIDFIELDER": "Centre Midfield", "RIGHTCENTREMIDFIELDER": "Centre Midfield",
    "LEFTCENTREMIDFIELDER": "Centre Midfield",
    "CENTREATTACKINGMIDFIELDER": "Number 8", "ATTACKINGMIDFIELDER": "Number 8",
    "RIGHTATTACKINGMIDFIELDER": "Number 8", "LEFTATTACKINGMIDFIELDER": "Number 8",
    "SECONDSTRIKER": "Number 8",
    "RIGHTWING": "Winger", "LEFTWING": "Winger",
    "RIGHTMIDFIELDER": "Winger", "LEFTMIDFIELDER": "Winger",
    "CENTREFORWARD": "Striker", "RIGHTCENTREFORWARD": "Striker", "LEFTCENTREFORWARD": "Striker",
}

def parse_first_position(cell):
    if pd.isna(cell):
        return ""
    return _clean_pos_token(str(cell))

def map_first_position_to_group(primary_pos_cell):
    tok = parse_first_position(primary_pos_cell)
    return RAW_TO_SIX.get(tok, None)

SIX_GROUPS = ["Full Back", "Centre Back", "Number 6", "Number 8", "Winger", "Striker"]

if "Primary Position" in df_all.columns:
    df_all["Six-Group Position"] = df_all["Primary Position"].apply(map_first_position_to_group)
else:
    df_all["Six-Group Position"] = np.nan

# Duplicate any "Centre Midfield" rows into both 6 and 8
if "Six-Group Position" in df_all.columns:
    cm_mask = df_all["Six-Group Position"] == "Centre Midfield"
    if cm_mask.any():
        cm_rows = df_all.loc[cm_mask].copy()
        cm_as_6 = cm_rows.copy(); cm_as_6["Six-Group Position"] = "Number 6"
        cm_as_8 = cm_rows.copy(); cm_as_8["Six-Group Position"] = "Number 8"
        df_all = pd.concat([df_all, cm_as_6, cm_as_8], ignore_index=True)

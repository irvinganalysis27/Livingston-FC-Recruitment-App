    # ---------- Title lines ----------
    top_parts = [str(player_name)]

    if role and isinstance(role, str):
        top_parts.append(str(role))
    if not pd.isnull(age):
        top_parts.append(f"{int(age)} years old")
    if not pd.isnull(height):
        top_parts.append(f"{int(height)} cm")

    # ensure all are strings
    top_parts = [str(x) for x in top_parts if x and str(x) != "nan"]
    line1 = " | ".join(top_parts)

    bottom_parts = []
    if team and isinstance(team, str):
        bottom_parts.append(team)
    if comp and isinstance(comp, str):
        bottom_parts.append(comp)
    if pd.notnull(mins):
        bottom_parts.append(f"{int(mins)} mins")
    if rank_val is not None:
        bottom_parts.append(f"Rank #{rank_val}")
    if score_100 is not None:
        bottom_parts.append(f"{score_100:.0f}/100")
    else:
        bottom_parts.append(f"Z {weighted_z:.2f}")

    bottom_parts = [str(x) for x in bottom_parts if x and str(x) != "nan"]
    line2 = " | ".join(bottom_parts)

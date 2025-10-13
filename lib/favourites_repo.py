from typing import Dict, Any, List, Optional
from .supabase_client import get_supabase

TABLE = "favourites"

def list_favourites(only_visible: bool = False) -> List[Dict[str, Any]]:
    sb = get_supabase()
    q = sb.table(TABLE).select("*")
    if only_visible:
        q = q.eq("visible", True)
    res = q.execute()
    return res.data or []

def get_favourites_map() -> Dict[str, Dict[str, Any]]:
    rows = list_favourites(only_visible=False)
    return {r["player"]: r for r in rows}

def upsert_favourite(
    player: str,
    team: str = "",
    league: str = "",
    position: str = "",
    colour: str = "",
    comment: str = "",
    visible: bool = True,
    updated_by: Optional[str] = None,
    source: Optional[str] = "dev"
) -> None:
    sb = get_supabase()
    payload = {
        "player": player,
        "team": team,
        "league": league,
        "position": position,
        "colour": colour,
        "comment": comment,
        "visible": visible,
        "updated_by": updated_by,
        "source": source,
    }
    sb.table(TABLE).upsert(payload, on_conflict="player").execute()

def set_visible(player: str, visible: bool) -> None:
    sb = get_supabase()
    sb.table(TABLE).update({"visible": visible}).eq("player", player).execute()

def delete_favourite(player: str) -> None:
    sb = get_supabase()
    sb.table(TABLE).delete().eq("player", player).execute()

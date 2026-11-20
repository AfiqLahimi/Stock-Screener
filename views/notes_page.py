"""Notes page: every research note you've written, searchable, grouped by stock."""
import streamlit as st

import ui
from core import display_name
from views.research_tabs import NOTE_TAGS, note_html


def render(C):
    notes = st.session_state.get("notes", {})
    if not notes:
        st.info("No notes yet. Open a stock in 📊 Analysis → 📝 Notes to write your thesis, plan or lessons.")
        return
    c1, c2 = st.columns([2, 1.4]) if not C.mobile else (st.container(), st.container())
    q = c1.text_input("Search", placeholder="Search your notes…", key="nt_q", label_visibility="collapsed").lower()
    tags = c2.multiselect("Tags", NOTE_TAGS, key="nt_tags", placeholder="All tags", label_visibility="collapsed")
    shown = 0
    for t, lst in sorted(notes.items(), key=lambda kv: kv[1][0]["date"] if kv[1] else "", reverse=True):
        sel = [n for n in lst if (not tags or n["tag"] in tags) and
               (not q or q in n["text"].lower() or q in display_name(t).lower() or q in t.lower())]
        if not sel:
            continue
        shown += len(sel)
        with ui.panel(f"{display_name(t)} · {t}", f"{len(sel)} note(s) · latest {sel[0]['date']}"):
            st.markdown("".join(note_html(n) for n in sel), unsafe_allow_html=True)
            st.button("📊 Open in Analysis", key=f"nt_open_{t}", on_click=ui.goto, args=("analysis",),
                      kwargs={"an_typed": t})
    if not shown:
        st.info("No notes match.")

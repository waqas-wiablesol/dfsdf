"""The standalone HTML coordinate check.

This is the only verification in the pipeline that has ever caught a real
problem. An in-bounds coordinate can still point at the wrong pixel, and no
arithmetic test finds that. Looking at the frames does.

The output is self-contained on purpose. Frame files live in a work directory
that gets cleaned up, so a report that only renders while that directory survives
is a report that will be lost. Images are embedded, which makes the file large
and permanent.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------
#
# These are plain constants with no substitution applied, which is the point.
# An earlier version built the whole page with `str.format`, so every CSS brace
# had to be doubled -- a rule that produces mangled styling when broken and gives
# no hint why. Concatenating unformatted constants and only formatting the small
# dynamic fragments removes that trap entirely.

_CSS = """\
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#14161a;color:#e6e6e6;margin:0;padding:18px}
h1{font-size:18px;font-weight:600;margin:0 0 4px}
.hint{color:#9aa0a6;font-size:13px;margin-bottom:14px;line-height:1.5;max-width:1100px}
.bar{position:sticky;top:0;background:#14161a;padding:10px 0;z-index:9;border-bottom:1px solid #2a2e35;margin-bottom:14px}
.filter{background:#22262d;color:#dfe3e6;border:1px solid #333941;border-radius:5px;padding:5px 11px;margin-right:6px;cursor:pointer;font-size:13px}
.filter.on{background:#00947e;border-color:#00b89c;color:#fff}
/* Two columns, not `auto-fill`. At 1920px an auto-fill with a 420px minimum
   yields FOUR narrow columns, which shrinks each frame to about a third of its
   native width -- too small to judge whether a coordinate is on the right
   control, which is the only thing this page exists to answer. Fixed at two so
   each frame stays large. The fallback is deliberately low (760px): a narrow
   window is common, and collapsing to one column at 1100px defeated the point
   of asking for two. */
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;align-items:start}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:#1b1e24;border:1px solid #2a2e35;border-radius:8px;overflow:hidden}
.wrap{position:relative;line-height:0;background:#000}
.wrap img{width:100%;display:block}
.dot{position:absolute;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;border:3px solid #ff3b30;box-sizing:border-box;box-shadow:0 0 0 1px rgba(0,0,0,.65),0 0 14px rgba(255,59,48,.8)}
.box{position:absolute;border:2px solid #00d1b2;box-sizing:border-box}
.meta{padding:10px 13px;font-size:13.5px;line-height:1.5}
.id{font-family:ui-monospace,Consolas,monospace;color:#8ab4f8}
.tag{background:#262b33;color:#b9c0c8;border-radius:3px;padding:1px 6px;font-size:11.5px;margin-left:4px}
.instr{color:#e6e6e6;margin:6px 0;font-size:14px}
.coord{color:#9aa0a6;font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.sub{color:#7d848c}
.empty{padding:40px;color:#9aa0a6}
"""

# No braces are substituted in here, so the JavaScript keeps its own syntax.
_SCRIPT = """\
document.querySelectorAll('.filter').forEach(function(btn){
  btn.addEventListener('click', function(){
    var want = btn.dataset.f;
    document.querySelectorAll('.filter').forEach(function(o){
      o.classList.toggle('on', o === btn);
    });
    document.querySelectorAll('.card').forEach(function(card){
      card.style.display = (want === '*' || card.dataset.t === want) ? '' : 'none';
    });
  });
});
"""

_HEADING = "Coordinate check"
_HINT = (
    "The red circle must sit on a real UI element in its own frame. "
    "The green box is the model's bounding box and must contain the circle. "
    "If the circle is consistently offset in one direction, or lands on empty "
    "canvas, the coordinates are wrong -- stop before using this dataset."
)


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def _box_style(bbox, width: int, height: int) -> str:
    """Inline style placing the bbox as a percentage overlay.

    Percentages rather than pixels, because the <img> is scaled to its column
    width and pixel offsets would drift with the layout.
    """
    if not bbox:
        return ""
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return ""
    return ('left:%.4f%%;top:%.4f%%;width:%.4f%%;height:%.4f%%;'
            % (100.0 * x1 / width, 100.0 * y1 / height,
               100.0 * max(0.0, x2 - x1) / width,
               100.0 * max(0.0, y2 - y1) / height))


def _card(row: dict) -> str | None:
    """One card, or None if its frame cannot be read.

    A missing frame is not an error here. The report exists to be *looked at*,
    and a card that cannot render is exactly the thing worth noticing, so it is
    reported by omission with a count rather than by crashing the report.
    """
    path = Path(row["image_path"])
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None

    width = max(1, int(row["image_width"]))
    height = max(1, int(row["image_height"]))
    x, y = int(row["target_x"]), int(row["target_y"])

    try:
        bbox = json.loads(row["bbox"])
    except (TypeError, ValueError):
        bbox = None

    box = _box_style(bbox, width, height)
    box_div = '<div class="box" style="%s"></div>' % box if box else ""
    answer = (row["answer_raw"]
              .replace("<coordinate>", "").replace("</coordinate>", "").strip())

    return (
        '<div class="card" data-t="{atype}">'
        '<div class="wrap">'
        '<img src="data:image/png;base64,{b64}" alt="{rid}">'
        "{box}"
        '<div class="dot" style="left:{px:.4f}%;top:{py:.4f}%"></div>'
        "</div>"
        '<div class="meta">'
        '<div class="id">{rid}<span class="tag">{atype}</span>'
        '<span class="tag">{t:.1f}s</span></div>'
        '<div class="instr">{instr}</div>'
        '<div class="coord">{w}x{h} &nbsp; abs ({x}, {y}) &nbsp; norm {ans}'
        '<br><span class="sub">{sub}</span></div>'
        "</div></div>"
    ).format(
        atype=html.escape(str(row["action_type"])),
        rid=html.escape(str(row["id"])),
        b64=encoded,
        box=box_div,
        px=100.0 * x / width,
        py=100.0 * y / height,
        t=float(row["timestamp_s"]),
        instr=html.escape(str(row["instruction"])),
        w=width, h=height, x=x, y=y,
        ans=html.escape(answer),
        sub=html.escape(str(row["sub_task"])[:150]),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render(rows: list[dict], out_path, limit: int = 200) -> Path:
    """Write the report. `limit=0` means every row.

    `limit` defaults to 200 because a 500-row report embeds 500 full-resolution
    PNGs and reaches tens of megabytes. Enough to judge, small enough to open.
    """
    picks = rows[:limit] if limit else rows
    out_path = Path(out_path)

    cards = []
    skipped = 0
    for row in picks:
        card = _card(row)
        if card is None:
            skipped += 1
        else:
            cards.append(card)

    action_types = sorted({str(r["action_type"]) for r in picks})
    buttons = "".join(
        '<button class="filter" data-f="%s">%s</button>'
        % (html.escape(a), html.escape(a)) for a in action_types)

    if cards:
        notice = ""
        if skipped:
            notice = (' <span class="sub">(%d card(s) omitted, frame '
                      "unreadable)</span>" % skipped)
        bar = ('<div class="bar">'
               '<button class="filter on" data-f="*">all (%d)</button>%s'
               "</div>" % (len(cards), buttons))
        grid = '<div class="grid">%s</div>' % "".join(cards)
    else:
        notice = ""
        bar = ""
        grid = ('<div class="empty">No cards could be rendered. Either no rows</div>')

    doc = (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<title>%s</title><style>%s</style></head><body>"
        "<h1>%s &mdash; %d rows%s</h1>"
        '<div class="hint">%s</div>'
        "%s%s"
        "<script>%s</script></body></html>"
        % (html.escape(_HEADING), _CSS, html.escape(_HEADING), len(cards),
           notice, _HINT, bar, grid, _SCRIPT)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path

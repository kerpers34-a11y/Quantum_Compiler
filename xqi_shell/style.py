from prompt_toolkit.styles import Style

style_html = Style.from_dict({
    'cg': '#00FF00',
    'cr': '#FF4500',
    'cy': '#FFD700',
    'cy2': '#B8860B',
    'ivory': '#FFFFF0',
    'cbg': 'bold #008000',
    'cbb': 'bold #1E90FF',
    'cir': 'italic #FF6100',
})

style_prompt = Style.from_dict({
    'pygments.keyword': '#F4A460',
    'pygments.literal.number': '#03A89E',
    'pygments.punctuation': '#B0C4DE',
    'pygments.comment': '#708090',

    # Prompt.
    "at":       "#00aa00",
    "colon":    "#00aa00",
    "pound":    "#00aa00",
    "host":     "#00ffff bg:#444400",
    "path":     "ansicyan underline",
})

message_prompt = [
    ("class:at",       "@"),
    ("class:host",     "localhost"),
    ("class:colon",    ":"),
    ("class:path",     "<XQI-QC> "),
]
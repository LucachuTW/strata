# Render the README GIF:  vhs demo.tape  ->  docs/strata-demo.gif
# Needs the stack up + corpus seeded first:
#   docker compose up -d && ollama serve & bash scripts/pull_models.sh
#   uv run strata seed-demo --reset
# Sleeps must exceed each command's runtime — tune to your model/GPU speed.
Output docs/strata-demo.gif

Set FontSize 18
Set Width 1280
Set Height 760
Set Theme "Dracula"
Set Padding 18

Type "uv run strata --version"   Enter   Sleep 3s

# The money shot: a restricted figure is refused at public clearance.
Type "uv run strata ask 'What is Acme Robotics restricted FY2026 revenue target?' --clearance public"
Enter   Sleep 30s

# Same fact, allowed at restricted clearance.
Type "uv run strata ask 'What is Acme Robotics annual revenue target for fiscal year 2026?' --clearance restricted"
Enter   Sleep 30s

# Multi-hop: CFO + risks, graph-expanded across documents.
Type "uv run strata ask 'Who is the CFO of Acme Robotics and which risks affect Acme Corporation?' --clearance restricted"
Enter   Sleep 35s

Sleep 3s

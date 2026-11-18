// ============================================================================
// Beginner-Friendly Step-by-Step Guide: Quartus + DE10-Standard Switches & GUI
// Compile with: typst compile fpga_easy_guide.typ fpga_easy_guide.pdf
// ============================================================================

#set document(
  title: "Step-by-Step Practical Guide: How to Use Quartus, Program the DE10-Standard Board, and Read Every Switch & Display",
  author: "Ersha Anandita",
)

#set page(
  paper: "a4",
  margin: (x: 1.8cm, y: 2cm),
  header: align(right)[
    #text(size: 8.5pt, fill: luma(110))[
      *Practical User Guide & Switch Cheat Sheet* • DE10-Standard FPGA CNN Accelerator
    ]
  ],
  footer: context [
    #line(length: 100%, stroke: 0.4pt + luma(180))
    #v(2pt)
    #grid(
      columns: (1fr, 1fr),
      align(left)[#text(size: 8.5pt, fill: luma(100))[Terasic DE10-Standard (`5CSXFC6D6F31C6`)]],
      align(right)[#text(size: 8.5pt, fill: luma(100))[Page #counter(page).display("1 of 1", both: true)]],
    )
  ],
)

#set text(
  font: ("Calibri", "Arial", "Times New Roman"),
  size: 10.5pt,
  lang: "en",
)

#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1.")

#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em)[
  #text(size: 14pt, weight: "bold", fill: rgb("#1e3a8a"))[#it]
  #v(-4pt)
  #line(length: 100%, stroke: 1.2pt + rgb("#1e3a8a"))
]

#let tipbox(title, body, color: rgb("#1d4ed8"), bg: rgb("#eff6ff")) = block(
  width: 100%,
  fill: bg,
  stroke: (left: 4pt + color, rest: 0.5pt + color.lighten(50%)),
  inset: 10pt,
  radius: (right: 4pt),
  above: 0.8em,
  below: 0.8em,
)[
  #text(weight: "bold", size: 11pt, fill: color)[#title] \
  #body
]

// ============================================================================
// TITLE BANNER
// ============================================================================
#align(center)[
  #block(
    width: 100%,
    fill: rgb("#0f172a"),
    inset: (x: 16pt, y: 18pt),
    radius: 6pt,
  )[
    #text(size: 10pt, weight: "bold", fill: rgb("#38bdf8"))[
      HANDS-ON BEGINNER GUIDE & HARDWARE CHEAT SHEET
    ]
    #v(5pt)
    #text(size: 16pt, weight: "bold", fill: white)[
      How to Use Your DE10-Standard FPGA Board, Every Switch & Display, Quartus Prime, and the Desktop GUI
    ]
    #v(6pt)
    #text(size: 10pt, fill: rgb("#cbd5e1"))[
      Plain-English Instructions for Running and Demonstrating Your Two-Layer CNN Image Classifier
    ]
  ]
]

// ============================================================================
// 1. WHAT IS HAPPENING ON YOUR FPGA BOARD RIGHT NOW?
// ============================================================================
= What Is Happening Inside Your DE10-Standard Board?

When we programmed your *DE10-Standard* board, we loaded two things into the Intel Cyclone V FPGA chip:
1. *10 Real Clothing Pictures* (one from each of the 10 Fashion-MNIST categories: `0 = T-shirt`, `1 = Trouser`, `2 = Pullover`, `3 = Dress`, `4 = Coat`, `5 = Sandal`, `6 = Shirt`, `7 = Sneaker`, `8 = Bag`, `9 = Ankle boot`), stored inside the FPGA's internal memory (`M10K ROM`).
2. *Your Two-Layer CNN Hardware Accelerator (`cnn_top.v`)*, which reads whichever picture you choose with the switches, runs all the convolution/pooling/classification math in hardware (`6.06 milliseconds`), and shows the answer on the *6 glowing red 7-segment number screens (`HEX5..HEX0`)* and *10 red LEDs (`LEDR9..LEDR0`)*.

#tipbox(
  "Why doesn't the FPGA board show a photograph on the board itself?",
  [
    The DE10-Standard board only has *7-segment number displays (`HEX5..HEX0`)*, *10 small red LED lights (`LEDR9..0`)*, *10 slide switches (`SW9..0`)*, and *4 pushbuttons (`KEY3..0`)*—it does not have an LCD screen attached. \
    - So on the *physical FPGA board*, you pick a picture number (`0` to `9`) using the switches, and the board shows you the *Class Number (`0` to `9`)* and *`P A S` (`PASS`)* on its 7-segment displays!
    - To see the *actual picture and confidence percentage `%`* on your laptop screen at the same time, you open our companion *Desktop GUI (`python -m sw.gui_app`)*!
  ],
)

// ============================================================================
// 2. MAP OF YOUR PHYSICAL DE10-STANDARD BOARD
// ============================================================================
= Physical Map of Your DE10-Standard Board

Look down at the bottom half of your physical *Terasic DE10-Standard* board. Here are the 4 parts we use:

#figure(
  block(
    width: 100%,
    fill: rgb("#f8fafc"),
    stroke: 1pt + rgb("#94a3b8"),
    inset: 12pt,
    radius: 5pt,
  )[
    #align(center)[
      #text(weight: "bold", size: 11pt, fill: rgb("#0f172a"))[
        PHYSICAL FRONT EDGE OF YOUR DE10-STANDARD BOARD
      ]
    ]
    #v(6pt)
    #table(
      columns: (1.3fr, 2.7fr),
      inset: 7pt,
      align: (left, left),
      fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else { white },
      [#text(fill: white, weight: "bold")[Board Part]],
      [#text(fill: white, weight: "bold")[Where It Is & What It Means]],
      [*`HEX5 HEX4 HEX3 HEX2 HEX1 HEX0`* \ *(6 Red Number Screens)*],
      [Located on the *bottom-left*. Reads from left (`HEX5`) to right (`HEX0`). \
      By default it shows: *`0 - P A S 0`* \
      • *Leftmost digit (`HEX5`)* = Which picture (`0..9`) you selected with the switches. \
      • *Middle letters (`HEX3..HEX1`)* = *`P A S`* (`PASS` — the FPGA guessed right!). \
      • *Rightmost digit (`HEX0`)* = *What class (`0..9`) the FPGA CNN predicted!*],
      [*`LEDR9 .. LEDR0`* \ *(10 Tiny Red LEDs)*],
      [Located *directly above the 10 slide switches*. \
      Whichever class (`0..9`) the FPGA predicts, *that exact LED lights up*! \
      • If Class `0` (`T-shirt`) wins → *`LEDR0`* (far right LED) turns ON. \
      • If Class `7` (`Sneaker`) wins → *`LEDR7`* turns ON.],
      [*`SW9 .. SW0`* \ *(10 Slide Switches)*],
      [Located along the *bottom edge*. \
      • *`SW0` is on the FAR RIGHT*. *`SW9` is on the FAR LEFT*. \
      • Pushed *DOWN* (toward you) = *`0` (OFF)*. \
      • Pushed *UP* (away from you) = *`1` (ON)*. \
      • The 4 rightmost switches (*`SW3, SW2, SW1, SW0`*) pick Picture `0` to `9` in binary! \
      • The leftmost switch (*`SW9`*) toggles the *Clock Cycle Counter (`302978`)*!],
      [*`KEY3 .. KEY0`* \ *(4 Blue/White Buttons)*],
      [Located on the *bottom-right corner*. \
      • *`KEY0` (Far-Right Button):* *Hardware Reset* (hold to reset, release to run). \
      • *`KEY1` (2nd Button from Right):* *Re-Run CNN* on the current picture. \
      • *`KEY2` (3rd Button from Right):* *Hardware Randomizer* (picks a random class `0..9`!). \
      • *`KEY3` (Far-Left Button):* *CLEAR DRAWING SCREEN* (wipes the `32×32` VGA canvas to black so you can draw from scratch with your PS/2 Mouse!).],
      [*`VGA OUT` & `PS/2 MOUSE`* \ *(Top Edge Connectors)*],
      [• *`VGA Port` (Blue 15-pin D-Sub):* Plug into any monitor for a live *`640×480 @ 60 Hz`* hardware dashboard showing the `$32 times 32$` canvas (`256×256` zoomed), mouse crosshair cursor, 10-class score bar chart, and winning class! \
      • *`PS/2 Port` (`PIN_AB25` / `PIN_AA25`):* Plug in a PS/2 mouse: *Left-Click + Drag* draws white strokes, *Right-Click + Drag* erases, and *Middle-Click* (or *`KEY3`*) clears the screen! As you draw, the FPGA classifies your drawing live in `6.06 ms`!],
    )
  ],
  caption: [How to locate and read the displays, LEDs, switches, buttons, VGA port, and PS/2 mouse on your DE10-Standard board.],
)

// ============================================================================
// 3. EXACT SWITCH CHEAT SHEET (HOW TO TEST ALL 10 CLOTHING PICTURES)
// ============================================================================
= Exact Switch Cheat Sheet: Test All 10 Clothing Pictures!

You use the *4 rightmost switches (`SW3, SW2, SW1, SW0`)* to choose which clothing image (`0` through `9`) the FPGA classifies. Every time you flip a switch, the FPGA *automatically* loads that picture from memory and runs the CNN in `6.06 milliseconds`!

*(Keep `SW9..SW4` pushed *DOWN* `▼` while testing these 10 pictures. `▲ UP` = `1`, `▼ DOWN` = `0`.)*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, 1.4fr, 1.3fr, auto, auto),
    inset: 5.5pt,
    align: (center, center, center, center, center, left, center, center, center),
    fill: (x, y) => if y == 0 { rgb("#1e3a8a") } else if calc.odd(y) { rgb("#f8fafc") } else { white },
    [#text(fill: white, weight: "bold")[Pic \#]],
    [#text(fill: white, weight: "bold")[`SW3`]],
    [#text(fill: white, weight: "bold")[`SW2`]],
    [#text(fill: white, weight: "bold")[`SW1`]],
    [#text(fill: white, weight: "bold")[`SW0`]],
    [#text(fill: white, weight: "bold")[Clothing Picture Inside FPGA]],
    [#text(fill: white, weight: "bold")[What `HEX5..HEX0` Shows]],
    [#text(fill: white, weight: "bold")[Lit LED]],
    [#text(fill: white, weight: "bold")[GUI Image \#]],
    [*`0`*], [▼ 0], [▼ 0], [▼ 0], [▼ 0], [*T-shirt/top* (Class `0`)], [*`0 - P A S 0`*], [`LEDR0`], [`Image 0`],
    [*`1`*], [▼ 0], [▼ 0], [▼ 0], [*▲ 1*], [*Trouser* (Class `1`)], [*`1 - P A S 1`*], [`LEDR1`], [`Image 10`],
    [*`2`*], [▼ 0], [▼ 0], [*▲ 1*], [▼ 0], [*Pullover* (Class `2`)], [*`2 - P A S 2`*], [`LEDR2`], [`Image 20`],
    [*`3`*], [▼ 0], [▼ 0], [*▲ 1*], [*▲ 1*], [*Dress* (Class `3`)], [*`3 - P A S 3`*], [`LEDR3`], [`Image 30`],
    [*`4`*], [▼ 0], [*▲ 1*], [▼ 0], [▼ 0], [*Coat* (Class `4`)], [*`4 - P A S 4`*], [`LEDR4`], [`Image 40`],
    [*`5`*], [▼ 0], [*▲ 1*], [▼ 0], [*▲ 1*], [*Sandal* (Class `5`)], [*`5 - P A S 5`*], [`LEDR5`], [`Image 50`],
    [*`6`*], [▼ 0], [*▲ 1*], [*▲ 1*], [▼ 0], [*Shirt* (Class `6`)], [*`6 - P A S 6`*], [`LEDR6`], [`Image 60`],
    [*`7`*], [▼ 0], [*▲ 1*], [*▲ 1*], [*▲ 1*], [*Sneaker* (Class `7`)], [*`7 - P A S 7`*], [`LEDR7`], [`Image 70`],
    [*`8`*], [*▲ 1*], [▼ 0], [▼ 0], [▼ 0], [*Bag* (Class `8`)], [*`8 - P A S 8`*], [`LEDR8`], [`Image 80`],
    [*`9`*], [*▲ 1*], [▼ 0], [▼ 0], [*▲ 1*], [*Ankle boot* (Class `9`)], [*`9 - P A S 9`*], [`LEDR9`], [`Image 90`],
  ),
  caption: [Exact switch combinations (`SW3 SW2 SW1 SW0`) for all 10 clothing classes and what your board displays.],
)

== Special Switch `SW9` (Far-Left Switch): View Hardware Speed (`302978` Cycles)
Want to prove to your professor/instructor that the FPGA hardware is genuinely counting clock cycles at `50 MHz`?
- Push *`SW9` (the far-left switch) UP (`▲ 1`)*:
  - The 6 red displays (`HEX5..HEX0`) immediately change from `0 - P A S 0` to:
    $ bold("3 0 2 9 7 8") $
  - That number (*`302,978`*) is the *exact number of 50 MHz clock cycles* the CNN took to finish classifying the image ($302,978 / 50,000,000 = 0.00606" seconds" = 6.06" ms"$)!
- Push *`SW9` back DOWN (`▼ 0`)* to return to the `0 - P A S 0` classification screen.

// ============================================================================
// 4. HOW TO USE QUARTUS PRIME (OPEN PROJECT, SEE REPORTS, & PROGRAM BOARD)
// ============================================================================
= Step-by-Step: How to Use Quartus Prime & Program the Board

*Note:* Because the FPGA uses volatile SRAM memory, *if you turn off the power button on the DE10-Standard board*, it forgets the program until you send the `.sof` file again. Here are two super easy ways to re-program it anytime:

== Way 1: The 1-Second PowerShell Command (Fastest)
Open your terminal in `c:\Users\Ersha Anandita\Downloads\fpga_midterm` and paste this single line:
```powershell
C:\intelFPGA_lite\18.1\quartus\bin64\quartus_pgm.exe -c "DE-SoC [USB-1]" -m JTAG -o "p;quartus/output_files/de10_standard.sof@2"
```
Within 3 seconds, your board lights up with `0 - P A S 0`!

== Way 2: Using the Quartus Prime Desktop App (Visual GUI)
If you want to show the project inside the *Quartus Prime 18.1* graphical app:
1. *Open the Project:*
   - Launch *Quartus Prime 18.1 Lite Edition* from the Windows Start Menu.
   - Click *File → Open Project...* and select:
     `C:\Users\Ersha Anandita\Downloads\fpga_midterm\quartus\de10_standard.qpf`
2. *Where to See the Synthesis & Resource Numbers in Quartus:*
   - Look at the center *Compilation Report* pane (or click *Processing → Compilation Report*).
   - Click *Fitter → Summary* to show your professor:
     - *Logic utilization (in ALMs):* `804 / 41,910 ( 2 % )`
     - *Total registers:* `646`
     - *Total RAM Blocks (M10K):* `31 / 553 ( 6 % )`
     - *Total DSP Blocks:* `22 / 112 ( 20 % )`
   - Click *Timing Analyzer → Slow 1100mV 85C Model → Fmax Summary* to show:
     - *Fmax:* `67.82 MHz` for `CLOCK_50` (with `+5.254 ns` positive setup slack).
3. *Where to See the Pin Assignments on the Chip:*
   - Click *Assignments → Pin Planner* in the top menu to see the visual Cyclone V chip map with `CLOCK_50`, `SW[9:0]`, `KEY[3:0]`, `LEDR[9:0]`, and `HEX0..HEX5` assigned to their physical pins.
4. *How to Program the Board from the Quartus GUI:*
   - Click *Tools → Programmer* in the top menu.
   - If the top-left *Hardware Setup...* box says `No Hardware`, click *Hardware Setup...*, select *`DE-SoC [USB-1]`* from the dropdown, and click *Close*.
   - Click *Auto Detect* (select `5CSXFC6D6F31`), right-click the second chip (`5CSXFC6D6F31`), click *Change File...*, and choose `quartus/output_files/de10_standard.sof`.
   - Check the *Program/Configure* box and click the *Start* button on the left. When the top-right progress bar says *`100% (Successful)`*, your board is live!

// ============================================================================
// 5. HOW TO RUN THE DESKTOP GUI ALONGSIDE YOUR FPGA BOARD
// ============================================================================
= How to Use the Desktop Picture GUI Alongside Your FPGA Board

To see the actual clothing image, the internal `Conv1 / MaxPool / Conv2 / GAP` feature maps, and the `10-class confidence percentages (%)` on your computer screen while flipping switches on the physical FPGA board:

1. Run this command in your terminal:
   ```powershell
   python -m sw.gui_app
   ```
2. In the top bar of the GUI window, use the *Jump to Category* dropdown (or type the Image Index from Table 2: `0, 10, 20, 30, 40, 50, 60, 70, 80, 90` and click *Go*):
   - When your FPGA board switches are set to *`0000` (`0`)* (`HEX5..0` = `0 - P A S 0`), look at *Image `0`* in the GUI: you see the exact `$32 times 32$` *T-shirt/top* picture, its internal feature maps, its winning hardware score (`+7`), and its *`82.4%` confidence*!
   - When your FPGA board switches are set to *`0111` (`7`)* (`HEX5..0` = `7 - P A S 7`), select *`7: Sneaker`* in the GUI dropdown: you see the exact *Sneaker* picture, its winning hardware score (`+34`), and its *`99.9%` confidence*!

#figure(
  image("reports/simulation/fpga_10class_detections.png", width: 100%),
  caption: [The exact 10 clothing pictures stored inside your DE10-Standard FPGA board (`SW[3:0] = 0` through `9`), showing the picture, the hardware `INT8` score, and the confidence percentage (`%`) for each switch setting.],
)

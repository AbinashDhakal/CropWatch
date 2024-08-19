# Stage 1 RPi demo instructions

## Preparation
On the host machine (Yavor Y's pc), a connection to the Pi is established using
<pre>
connect-rpi</pre>
On the RPi the demo is prepared using
<pre>
cd RPiTest
demo-prepare
</pre>

The second command mounts the SD card as read-write and cleans the output folder.

## Demonstrate input and output folder
Input folder contents are listed using
<pre>
ls Test
</pre>

Output is demonstrated to be empty with
<pre>
ls Filtered_Image
</pre>

## Run the demo
<pre>
demo-run
</pre>
This command is an alias for running the python script with the correct virtual environment.

## Demonstrate output
<pre>
ls Filtered_Image
</pre>

## Downlink folder
Demonstrate through the GUI interface that the `RPI_DOWNLINK` folder is empty.

## Downlink
The alias of the full `scp` command is
<pre> downlink
</pre>

## Show files in `RPI_DOWNLINK`

# Stage 2 demo instructions

## Navigate to the correct directory and clean the output folder

<pre>cd /home/yavor/Documents/sc/STAGE2_DEMO
demo2-prepare </pre>

## Show `OUTPUT`

## Run the demo
<pre> demo2-run  </pre>

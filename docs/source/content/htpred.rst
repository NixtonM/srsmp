..    include:: <isonum.txt>

Application Use
===============

**SRSMP** prediction application script is implemented to be used in conjungtion with 
*SpatialAnalyzer*. The communication between the two applications is implemented using 
the **datashare** file format. This *How To* shows the step-by-step instructions on how 
to setup and use the framework. 

To minimize time needed per measurement, the python side of the application framework was
implemented in a server/client architecture. A local server maintains the connection with 
the *Thorlabs spectrometer*. The client sends the appropiate commands to the server to 
trigger measurements etc.

Following this *How To*, the user will be able to get a material property prediciton 
coupled with the corresponding 3D coordinates within SpatialAnalyzer.

Python Set-Up
-------------
Copy the final trained model and ini file from the results directory to the model directory 
(:file:`model_dir`), as defined in the config file.

Run :file:`repo/scripts/03a_application_server.py` to start the local server and wait for 
the message *'Ready for reference measurement'*. Open a separate command prompt and while 
holding the **probe tip** to the reference spectra and run:

.. code-block:: shell

   $ cd repo/scripts
   $ python 03b_application_client.py -1

This measures the spectrum of the reference spectra and saves it for the following predictions. 
The argument `-1` must be replaced by `-2` if using Spectralon 25% instead of 60%.

The server is now ready for the measurements to be triggered from SpatialAnalyzer (via the 
client).

SpatialAnalyzer Set-Up
----------------------
To attain the desired behaviour, we use *measurement plans (MP)* and *MP execution 
nodes*. The appropiate *MP* can be found in :file:`repo/spatialanalyzer_mp/thor_predictor.mp`.

Open the *MP editor* by clicking on :guilabel:`Scripts` |rarr| :guilabel:`Create/Edit 
Measurement Plan`. Load the previously mentioned MP and change the value of *A0* in *Step 
0* to match the path defined as *base_dir* in the :file:`config.ini`. Do **not** change any 
other definitions. Save the changes to the *MP* and close the editor.

Embed the *MP* by clicking on :guilabel:`Scripts` |rarr| :guilabel:`Embedded Measurement Plans` 
|rarr| :guilabel:`Embed existing .MP file` and select your saved *MP*.

Add a *MP execution node* by clicking :guilabel:`Relationship` |rarr| :guilabel:`MP Execution 
Node` and assign the *embedded MP* by expanding *Measurement Plans* in the *TreeBar* on the left, 
right-clicking your *MP* and selecting :guilabel:`Add To Relationship`. In the newly opened 
*Relationship Selection* window, select *MP Execution* and click :guilabel:`OK`.

Enable the *Toolkit toolbar* by clicking :guilabel:`View` |rarr| :guilabel:`Show Toolkit Bar` 
and select the :guilabel:`Inspection` tab. *MP Execution* should be listed solely 
under *Task*. If there are additional *inspection tasks*, be aware that these will alternate 
with the *spectral measurement* on each button click. We therefore recommend removing all 
other tasks.

Connect the *Leica AT960/930* and then run the instrument interface. By having the *MP Execution* 
trapped to *inspections*, the *MP* can be triggered sending the *Next* command from the instrument. 
This command is by default assigned to **C button** on the *T-Probe* but can be reassigned under 
:guilabel:`Settings` |rarr| :guilabel:`Tracker` |rarr| :guilabel:`General Settings` |rarr| 
:guilabel:`Leica AT960/930` |rarr| :guilabel:`Program Buttons`.



Once all measurements have been taken, run in the command prompt:

.. code-block:: shell

   $ python 03b_application_client.py -99

This will save the spectral measurements and shut down the server.
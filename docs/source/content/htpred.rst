..    include:: <isonum.txt>

Application Use
===============

**SRSMP** prediction application script is implemented to be used in conjungtion with 
*SpatialAnalyzer*. The communication between the two applications is implemented using 
the **datashare** file format. This *How To* shows the step-by-step instructions on how 
to setup and use the framework. For a more in depth look at the implementation, please 
consult **LINK**.

Following this *How To*, the user will be able to get a material property prediciton 
coupled with the corresponding 3D corrdinates within SpatialAnalyzer.

Set-Up SpatialAnalyzer
----------------------
To attain the desired behaviour, we use *measurement plans (MP)* and *MP execution 
nodes*. The appropiate *MP* can be found in :file:`REPO/spatialanalyzer_mp/thor_predictor.mp`.

Open the *MP editor* by clicking on :guilabel:`Scripts` |rarr| :guilabel:`Create/Edit 
Measurement Plan`. Load the previously mentioned MP and change the value of *A0* in *Step 
0* to match the path defined as *base_dir* in the :file:`config.ini`. Do **not** change any 
other definitions. Save the changes to the *MP* and close the editor.

Embed the *MP* by clicking on :guilabel:`Scripts` |rarr| :guilabel:`Embedded Measurement Plans` 
|rarr| :guilabel:`Embed existing .MP file` and select your saved *MP*.

Add a *MP execution node* by clicking :guilabel:`Relationship` |rarr| :guilabel:`MP Execution 
Node` and assign the *embedded MP* by expaning *Measurement Plans* in the *TreeBar* on the left, 
right-clicking your *MP* and selecting :guilabel:`Add To Relationship`. In the newly openend 
*Relationship Selection* window, select *MP Execution* and click :guilabel:`OK`.

Enable the *Toolkit toolbar* by clicking :guilabel:`View` |rarr| :guilabel:`Show Toolkit Bar` 
and select the :guilabel:`Inspection` tab. *MP Execution* should be listed solely listed 
under *Task*. If there are additional *inspection tasks*, be aware that these will alternate 
with the *spectral measurement* on each button click. We therefore recommend removing all 
other tasks.

Connect the *Leica AT960/930* and the run the instrument interface. By having the *MP Execution* 
trapped to *inspections*, the *MP* can be triggered sending the *Next* command. This command 
is by default assigned to **C button** on the *T-Probe* but can be reassigned under 
:guilabel:`Settings` |rarr| :guilabel:`Tracker` :guilabel:`General Settings` |rarr| 
:guilabel:`Leica AT960/930` |rarr| :guilabel:`Program Buttons`.

SpatialAnalyzer
---------------
asd

Tool Offset
-----------
asd
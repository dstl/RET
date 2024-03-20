var CanvasContinuousModule = function (canvas_width, canvas_height, canvas_background_path) {
	// Create the element
	// ------------------

	// Create the tag with absolute positioning :
	var canvas_tag = `<canvas width="${canvas_width}" height="${canvas_height}" class="world-grid"`;
	canvas_tag += `style='background-repeat: no-repeat; background-size:100% 100%; background-color:transparent'></canvas>`;

	var parent_div_tag = `<div style='background-image: url("${canvas_background_path}"); height:${canvas_height}px;`;
	parent_div_tag += `width:${canvas_width}px; background-repeat: no-repeat; background-size:100% 100%;' class='world-grid-parent'></div>`;

	// Append it to body:
	var canvas = $(canvas_tag)[0];
	var interaction_canvas = $(canvas_tag)[0];
	var parent = $(parent_div_tag)[0];

	//$("body").append(canvas);
	$("#elements").append(parent);
	parent.append(canvas);
	parent.append(interaction_canvas);

	// Create the context and the drawing controller:
	var context = canvas.getContext("2d");

	var interactionHandler = new InteractionHandlerCont(canvas_width, canvas_height, interaction_canvas.getContext("2d"));
	var canvasDraw = new ContinuousVisualization(canvas_width, canvas_height, context, interactionHandler);

	this.render = function (data) {
		canvasDraw.resetCanvas();
		if (interactionHandler)
			interactionHandler.mouseoverLookupTable.reset();
		for (var layer in data) {
			canvasDraw.drawLayer(data[layer]);
		}
	};

	this.reset = function () {
		canvasDraw.resetCanvas();
	};

};

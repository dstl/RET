/**
Mesa Canvas Continuous Visualization
====================================================================

This is JavaScript code to visualize a Mesa Continuous state using the
HTML5 Canvas. Here's how it works:

On the server side, the model developer will have assigned a portrayal to each
agent type. The visualization then loops through the continuous space, for each object adds
a JSON object to an inner list to be sent to the
browser.

Each JSON object to be drawn contains the following fields: Shape (currently
only rectangles, circles and custom logos [images] are supported), x, y,
Color, Filled (boolean); circles also get a Radius, while rectangles get x and y sizes.
The rectangle  values are all between [0, 1] and get scaled to the space size.

The browser (this code, in fact) then draws them in.

Here's a sample input:

{"Shape": "rect", "x": 0, "y": 0, "w": 0.5, "h": 0.5, "Color": "#00aa00", "Filled": "true"}
*/

var ContinuousVisualization = function (width, height, context, interactionHandler) {

	var height = height;
	var width = width;
	var context = context;
	var defaultIconWidth = 20
	var defaultIconHeight = 20

	this.drawLayer = function (portrayalLayer) {
		(interactionHandler) ? interactionHandler.mouseoverLookupTable.init() : null
		for (var i in portrayalLayer) {
			var p = portrayalLayer[i];
			(interactionHandler) ? interactionHandler.mouseoverLookupTable.set(p) : null
			if (!p.stroke_color)
				if(p.Color)
					p.stroke_color = p.Color[0]
				else
					p.stroke_color = '#000000'

			// Does the inversion of y positioning because of html5
			// canvas y direction is from top to bottom. But we
			// normally keep y-axis in plots from bottom to top.
			p.y = 1 - p.y;

			if (p.Shape == "rect")
				this.drawRectangle(p.x, p.y, p.w, p.h, p.Color, p.stroke_color, p.Filled, p.text, p.text_color);
			else if (p.Shape == "circle")
				this.drawCircle(p.x, p.y, p.r, p.Color, p.stroke_color, p.Filled, p.text, p.text_color);
			else if (p.Shape == "svg")
				this.drawSVG(p.svgSource, p.x, p.y, p.scale, p.text, p.text_color)
			else
				this.drawCustomImage(p.Shape, p.x, p.y, p.scale, p.text, p.text_color)
		};
		// if a handler exists, update its mouse listeners with the new data
		(interactionHandler) ? interactionHandler.updateMouseListeners(portrayalLayer): null;
	};

	// DRAWING METHODS
	// =====================================================================

	/**
	Draw a circle.
	x, y: Space coords
	r: Radius relative to the default icon size ({1} would give a circle with the same height as the default icon size)
	colors: List of colors for the gradient. Providing only one color will fill the shape with only that color, not gradient.
	stroke_color: Color to stroke the shape
	fill: Boolean for whether or not to fill the circle.
	text: Inscribed text in rectangle.
	text_color: Color of the inscribed text.
	*/
	this.drawCircle = function (x, y, radius, colors, stroke_color, fill, text, text_color) {
		var cx = (x * width) + defaultIconWidth/2;
		var cy = (y * height) + defaultIconHeight/2;
		var r = radius * defaultIconHeight;

		context.beginPath();
		context.arc(cx, cy, r, 0, Math.PI * 2, false);
		context.closePath();

		context.strokeStyle = stroke_color;
		context.stroke();

		if (fill) {
			var gradient = context.createRadialGradient(cx, cy, r, cx, cy, 0);

			for (i = 0; i < colors.length; i++) {
				gradient.addColorStop(i/colors.length, colors[i]);
			}

			context.fillStyle = gradient;
			context.fill();
		}

		// This part draws the text inside the Circle
		if (text !== undefined) {
			context.fillStyle = text_color;
			context.textAlign = 'center';
			context.textBaseline = 'middle';
			context.fillText(text, cx, cy);
		}

	};

	/**
	Draw a rectangle in the specified grid cell.
	x, y: Positional coordinates as a fraction of the space [0,1] which represent the centre point of the rectangle.
	w, h: Width and height relative to the default icon size ({1,1} would give a rectangle the same size as the default icon size).
	colors: List of colors for the gradient. Providing only one color will fill the shape with only that color, not gradient.
	stroke_color: Color to stroke the shape
	fill: Boolean, whether to fill or not.
	text: Inscribed text in rectangle.
	text_color: Color of the inscribed text.
	*/
	this.drawRectangle = function (x, y, w, h, colors, stroke_color, fill, text, text_color) {
		context.beginPath();
		var dx = w * defaultIconWidth;
		var dy = h * defaultIconHeight;

		// Keep the drawing centered:
		var x0 = (x * width) - 0.5 * dx;
		var y0 = (y * height) - 0.5 * dy;

		context.strokeStyle = stroke_color;
		context.strokeRect(x0, y0, dx, dy);

		if (fill) {
			var gradient = context.createLinearGradient(x0, y0, x0 + cellWidth, y0 + cellHeight);

			for (i = 0; i < colors.length; i++) {
				gradient.addColorStop(i/colors.length, colors[i]);
			}

			// Fill with gradient
			context.fillStyle = gradient;
			context.fillRect(x0,y0,dx,dy);
		}
		else {
			context.fillStyle = color;
			context.strokeRect(x0, y0, dx, dy);
		}
		// This part draws the text inside the Rectangle
		if (text !== undefined) {
			var cx = x * width;
			var cy = y * height;
			context.fillStyle = text_color;
			context.textAlign = 'center';
			context.textBaseline = 'middle';
			context.fillText(text, cx, cy);
		}

	};

	/**
	Draw an svg on the canvas.
	x, y: Positional coordinates as a fraction of the space [0,1] which represent the centre point of the image/icon.
	scale: Scaling of the custom image [0, 1]
	text: Inscribed text in shape.
	text_color: Color of the inscribed text.
	*/
	this.drawSVG = function (svgSource, x, y, scale, text, text_color) {
		let blob = new Blob([svgSource], {type: 'image/svg+xml'});
		let url = URL.createObjectURL(blob);
		let image = document.createElement('img');
		image.src = url;
		this.drawImage(image, x, y, scale, text, text_color)
	};

	/**
	Draw a custom image on the canvas.
	x, y: Positional coordinates as a fraction of the space [0,1] which represent the centre point of the image/icon.
	scale: Scaling of the custom image [0, 1]
	text: Inscribed text in shape.
	text_color: Color of the inscribed text.
	*/
	this.drawCustomImage = function (shape, x, y, scale, text, text_color) {
		var img = new Image();
		img.src = "local/".concat(shape);
		this.drawImage(img, x, y, scale, text, text_color)
	}

	/**
	Draw an image on the canvas.
	img: HTML Image Element to be drawn on the context.
	x, y: Positional coordinates as a fraction of the space [0,1] which represent the centre point of the image/icon.
	scale: Scaling of the custom image [0, 1]
	text: Inscribed text in shape.
	text_color: Color of the inscribed text.
	*/
	this.drawImage = function (img, x, y, scale, text, text_color) {
		if (scale === undefined) {
			var scale = 1
		}

		// Coordinates of the central point
		var x0 = x * width;
		var y0 = y * height;

		img.onload = function () {
			var dHeight = defaultIconHeight * scale;
			var aspectRatio = this.width / this.height
			var dWidth = dHeight * aspectRatio;

			// Corner coordinates as the image is drawn from the corner
			var cx = x0 - (dWidth/2);
			var cy = y0 - (dHeight/2);

			context.drawImage(img, cx, cy, dWidth, dHeight);
			// This part draws the text on the image
			if (text !== undefined) {
				// ToDo: Fix fillStyle
				// context.fillStyle = text_color;
				context.textAlign = 'center';
				context.textBaseline = 'middle';
				context.fillText(text, cx, cy);
			}
		}
	};

	this.resetCanvas = function () {
		context.clearRect(0, 0, height, width);
		context.beginPath();
	};

};

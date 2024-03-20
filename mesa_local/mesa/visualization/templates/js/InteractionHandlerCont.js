
/**
Mesa Visualization InteractionHandler
====================================================================

This uses the context of an additional canvas laid overtop of another canvas
visualization and maps mouse movements to agent position, displaying any agent
attributes included in the portrayal that are not listed in the ignoredFeatures.

The following portrayal will yield tooltips with wealth, id, and pos:

portrayal = {
   "Shape": "circle",
   "Filled": "true",
   "Layer": 0,
   "Color": colors[agent.wealth] if agent.wealth < len(colors) else '#a0a',
   "r": 0.3 + 0.1 * agent.wealth,
   "wealth": agent.wealth,
   "id": agent.unique_id,
   'pos': agent.pos
}

**/

var InteractionHandlerCont = function(width, height, ctx){

  const lineHeight = 10;

	// list of standard rendering features to ignore (and key-values in the portrayal will be added )
	const ignoredFeatures = [
		'Shape',
		'svgSource',
		'Filled',
		'Color',
		'r',
		'x',
		'y',
    'w',
    'h',
		'width',
		'height',
    'heading_x',
    'heading_y',
		'stroke_color',
    'text_color'
	];


  //Set up a table of agents and their positions
  var mouseoverLookupTable = this.mouseoverLookupTable = buildMouseoverLookupTable();
  function buildMouseoverLookupTable(){
    var lookupTable
    this.init = function(){
      (lookupTable) ? null : lookupTable = [];
    }

    this.reset = function(){
      lookupTable = null
    }

    this.set = function(p){
      lookupTable.push(p)
    }

    this.get = function(x, y){
      var nearAgents = []
      lookupTable.forEach(p => {
        var xs = p.x - x
        var ys = (1 - p.y) - y
        xs *= xs
        ys *= ys
        dist = Math.sqrt(xs + ys)
        if(dist < 0.03){
          nearAgents.push(p)}
      })
      return nearAgents
    }

    return this;
  }


  // wrap the rect styling in a function
  function drawTooltipBox(ctx, x, y, width, height){
    ctx.fillStyle = "#F0F0F0";
    ctx.beginPath();
    ctx.shadowOffsetX = -3;
    ctx.shadowOffsetY = 2;
    ctx.shadowBlur = 6;
    ctx.shadowColor = "#33333377";
    ctx.rect(x, y, width, height);
    ctx.fill();
    ctx.shadowColor = "transparent";
  }

  var listener; var tmp
  this.updateMouseListeners = function(portrayalLayer){tmp = portrayalLayer


      // Remove the prior event listener to avoid creating a new one every step
      ctx.canvas.removeEventListener("mousemove", listener);
      // define the event listener for this step
      listener = function(event){
  			// clear the previous interaction
        ctx.clearRect(0, 0, width, height);

        // map the event to normalised x,y coordinates with bottom left as origin
        const yPosition = 1 - event.offsetY/height;
        const xPosition = event.offsetX/width;
        const position = {xPosition, yPosition};


        // look up the portrayal items the coordinates refer to and draw a tooltip
        mouseoverLookupTable.get(position.xPosition, position.yPosition).forEach((portrayal, nthAgent) => {
            const agent = portrayal;
            const features = Object.keys(agent).filter(k => ignoredFeatures.indexOf(k) < 0);
            const textWidth = Math.max.apply(null, features.map(k => ctx.measureText(`${k}: ${agent[k]}`).width));
  					const textHeight = features.length * lineHeight;
  					const y = Math.max(lineHeight * 2, Math.min(height - textHeight, event.offsetY - textHeight/2));
            const rectMargin = 2 * lineHeight;
            var x = 0;
            var rectX = 0;

            if(event.offsetX < width/2){
              x = event.offsetX + rectMargin + nthAgent * (textWidth + rectMargin);
              ctx.textAlign = "left";
              rectX = x - rectMargin/2;
            } else {
              x = event.offsetX - rectMargin - nthAgent * (textWidth + rectMargin + lineHeight );
              ctx.textAlign = "right";
              rectX = x - textWidth - rectMargin/2;
            }

            // draw a background box
            drawTooltipBox(ctx, rectX, y - rectMargin, textWidth + rectMargin, textHeight + rectMargin);

            // set the color and draw the text
            ctx.fillStyle = "black";
  					features.forEach((k,i) => {
              ctx.fillText(`${k}: ${agent[k]}`, x, y + i * lineHeight)
            })
        })

    };
    ctx.canvas.addEventListener("mousemove", listener);
  };

  return this;
}

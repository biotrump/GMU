#include "sdlwrapper.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>


#ifdef _WIN32
#include <windows.h>

/**
 * Get the current time, from the first call
 * of this function
 * @return The time in seconds
 */
float getTime(void){

	static float startTime = -1;
    SYSTEMTIME st;

    GetLocalTime(&st);
    float seconds = st.wHour*3600.0f + st.wMinute*60.0f 
		+ st.wSecond*1.0f + (st.wMilliseconds)/1000.0f;
	if (startTime < 0) {
		startTime = seconds;
	}

    return seconds - startTime;
}
#else 
#include <sys/timeb.h>

/**
 * Get the current time, from the first call
 * of this function
 * @return The time in seconds
 */
float getTime(void){
    static time_t start = -1;
    struct timeb currentTime;
    //struct timezone tzz;
    ftime(&currentTime);
    if(start < 0){
	start = currentTime.time;
    }
    return (float)(currentTime.time-start) + 
        (float)(currentTime.millitm)/1000.0f;
}
#endif

/**
 * Print a message to the stdout
 * @param debugLevelThe seriousness of the log
 * @param fmt Format string - same as printf
 */
void logMessage(int debugLevel, const char* fmt, ...){
	va_list ap;
	va_start(ap, fmt);
	switch(debugLevel){
		case DEBUG_LEVEL_ERROR:
			fprintf(stderr, "ERROR: ");
			vfprintf(stderr, fmt, ap);
			fprintf(stderr, "\n");
			break;
#ifdef DEBUG
		case DEBUG_LEVEL_WARNING:
			fprintf(stdout, "WARNING: ");
			vfprintf(stdout, fmt, ap);
			fprintf(stdout, "\n");
			break;
		default:
			fprintf(stdout, "LOG: ");
			vfprintf(stdout, fmt, ap);
			fprintf(stdout, "\n");
			break;
#endif
	}
	va_end(ap);
}


/**
 * Initialize the output framebuffer
 * @param width The width of the screen
 * @param height The height of the screen
 * @param color SDL attribute
 * @return The created sdl surface
 */
SDL_Surface * initScreen(unsigned int width, unsigned int height, unsigned int color) {

	// Create window
	SDL_Surface * screen = SDL_SetVideoMode(width, height, color, SDL_DOUBLEBUF | SDL_RESIZABLE);
	if(screen == NULL) throw SDL_Exception();

	return screen;
}

int readImage(const char* name, SDL_Surface **image){
    SDL_Surface *temp = IMG_Load(name);
    
    if(temp == NULL) {
        printf("Unable to load bitmap %s\n.", name);
        printf("Reason: %s ", SDL_GetError());
        return -1;
    }
    
    SDL_PixelFormat format;
    format.BitsPerPixel = 32;
    format.BytesPerPixel = 4;
    format.palette = NULL;
    format.Rmask = 0xff; format.Rshift = 0; format.Rloss = 0;
    format.Gmask = 0xff00; format.Gshift = 8; format.Gloss = 0;
    format.Bmask = 0xff0000; format.Bshift = 16; format.Bloss = 0;
    format.Amask = 0xff000000; format.Ashift = 24; format.Aloss = 0;
    format.colorkey = 0x00000000;
    format.alpha = 0xff;
    *image = SDL_ConvertSurface(temp, &format, SDL_SWSURFACE);
    SDL_FreeSurface(temp);
    return 0;
}

/**
 * The mainloop for the application using SDL
 */
void mainLoop(SDL_Surface *screen){
	// Call init code
	onInit();

	onWindowResized(screen->w, screen->h);

	// Window is not minimized
	bool active = true;

	for(;;)// Infinite loop
	{
		SDL_Event event;

		// Wait for event
		if(SDL_WaitEvent(&event) == 0) throw SDL_Exception();

		// Screen needs redraw
		bool redraw = false;

		// Handle all waiting events
		do
		{
			// Call proper event handlers
			switch(event.type)
			{
				case SDL_ACTIVEEVENT :// Stop redraw when minimized
					if(event.active.state == SDL_APPACTIVE)
						active = (bool) event.active.gain;
					break;
				case SDL_KEYDOWN :
					onKeyDown(event.key.keysym.sym, event.key.keysym.mod);
					break;
				case SDL_KEYUP :
					onKeyUp(event.key.keysym.sym, event.key.keysym.mod);
					break;
				case SDL_MOUSEMOTION :
					onMouseMove(event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel, event.motion.state);
					break;
				case SDL_MOUSEBUTTONDOWN :
					onMouseDown(event.button.button, event.button.x, event.button.y);
					break;
				case SDL_MOUSEBUTTONUP :
					onMouseUp(event.button.button, event.button.x, event.button.y);
					break;
				case SDL_QUIT :
					return;// End main loop
				case SDL_VIDEORESIZE :
					onWindowResized(event.resize.w, event.resize.h);
					break;
				case SDL_VIDEOEXPOSE :
					redraw = true;
					break;
				default :// Do nothing
					break;
			}
		} while(SDL_PollEvent(&event) == 1);

		// Optionally redraw window
		if(active && redraw) onWindowRedraw();
	}
}

/**
 * mainloop with timer generating redraw events ever period ms
 * @param period
 * @param screen The output framebuffer
 */
void mainLoop(unsigned period, SDL_Surface *screen) {
	// Call init code
	onInit();

	onWindowResized(screen->w, screen->w);

	// This main loop requires timer support
	if(SDL_InitSubSystem(SDL_INIT_TIMER) < 0) throw SDL_Exception();

	// Create redraw timer
	class RedrawTimer
	{
		private :
			SDL_TimerID id;
			static Uint32 callback(Uint32 interval, void *)
			{
				redraw();
				return interval;
			}
		public :
			RedrawTimer(unsigned interval)
			: id(SDL_AddTimer(interval, callback, NULL))
			{
				if(id == NULL) throw SDL_Exception();
			}
			~RedrawTimer()
			{
				if(id != NULL) SDL_RemoveTimer(id);
			}
	} redrawTimer(period);

	// Window is not minimized
	bool active = true;

	for(;;)// Infinite loop
	{
		SDL_Event event;

		// Wait for event
		if(SDL_WaitEvent(&event) == 0) throw SDL_Exception();

		// Screen needs redraw
		bool redraw = false;

		// Handle all waiting events
		do
		{
			// Call proper event handlers
			switch(event.type)
			{
				case SDL_ACTIVEEVENT :// Stop redraw when minimized
					if(event.active.state == SDL_APPACTIVE)
					active = event.active.gain;
					break;
				case SDL_KEYDOWN :
					onKeyDown(event.key.keysym.sym, event.key.keysym.mod);
					break;
				case SDL_KEYUP :
					onKeyUp(event.key.keysym.sym, event.key.keysym.mod);
					break;
				case SDL_MOUSEMOTION :
					onMouseMove(event.motion.x, event.motion.y, event.motion.xrel, event.motion.yrel, event.motion.state);
					break;
				case SDL_MOUSEBUTTONDOWN :
					onMouseDown(event.button.button, event.button.x, event.button.y);
					break;
				case SDL_MOUSEBUTTONUP :
					onMouseUp(event.button.button, event.button.x, event.button.y);
					break;
				case SDL_QUIT :
					return;// End main loop
				case SDL_VIDEORESIZE :
					onWindowResized(event.resize.w, event.resize.h);
					break;
				case SDL_VIDEOEXPOSE :
					redraw = true;
					break;
				default :// Do nothing
					break;
			}
		} while(SDL_PollEvent(&event) == 1);

		// Optionally redraw window
		if(active && redraw) onWindowRedraw();
	}
}

#ifndef _SDLWRAP_H_
#define _SDLWRAP_H_

#include <exception>
#include <stdexcept>
#include <string>

#include <SDL/SDL.h>
#include <SDL/SDL_image.h>

/**
 * Miscellenous functions
 */
//#define DEBUG

enum {
	DEBUG_LEVEL_WARNING = 0,
	DEBUG_LEVEL_ERROR,
	DEBUG_LEVEL_LOG
};

/**
 * Print a message to stdout/stderr
 * @param debugLevel Level of debugging
 * @param msg The message that should be printed
 */
void logMessage(int debugLevel, const char* fmt, ...);

/**
 * Float to string
 */
inline std::string ftos(float value, bool precise = false){
	char buffer[32];
	if(precise){
		sprintf(buffer, "%.5f", value);
	} else {
		sprintf(buffer, "%.1f", value);
	}
	buffer[31] = '\0';
	return std::string(buffer);
}

/**
 * Float to string
 */
inline std::string itos(int value){
	char buffer[32];
	sprintf(buffer, "%i", value);
	buffer[31] = '\0';
	return std::string(buffer);
}

/**
 * Check for an OpenGL error.
 * If error was found throw an SDL_Exception and quit.
 */
void glCheckError(const char * title);

/**
 * Return the number of seconds from the first call
 */
float getTime();

/**
 *SDL wrapper functions
 */

int readImage(const char* name, SDL_Surface **image);

/**
 * Exception class for sdl
 */
struct SDL_Exception : public std::runtime_error
{
    SDL_Exception() throw() : std::runtime_error(std::string("SDL : ") + SDL_GetError()) {}
};


/**
 * Send a quit event to SDL mainloop
 */
inline void quit()
{
    SDL_Event event;
    event.type = SDL_QUIT;
    if(SDL_PushEvent(&event) < 0) throw SDL_Exception();
}

/**
 * Send a redraw event to SDL mainloop
 */
inline void redraw()
{
    SDL_Event event;
    event.type = SDL_VIDEOEXPOSE;
    if(SDL_PushEvent(&event) < 0) throw SDL_Exception();
}

/**
 * Handlers for events called from SDL mainloop
 */

/**
 * Called after context was created
 */
void onInit();

/**
 * Called when the window should be redrawn
 */
void onWindowRedraw();

/**
 * Called when the window was resized
 * @param width The new width
 * @param height The new height
 */
void onWindowResized(int width, int height);

/**
 * Called when the key was pressed
 * @param key The key that was pressed
 * @param mod Modifiers
 */
void onKeyDown(SDLKey key, SDLMod mod);

/**
 * Called when the key was released
 * @param key The key that was released
 * @param mod Modifiers
 */
void onKeyUp(SDLKey key, SDLMod mod);

/**
 * Called when the mouse moves over the window
 * @param x The new x position
 * @param y The new y position
 * @param xrel Relative move from last x position
 * @param yrel Relative move from last y position
 * @param buttons Mask of the buttons that are pressed
 */
void onMouseMove(unsigned x, unsigned y, int xrel, int yrel, Uint8 buttons);

/**
 * Called when a mouse button was pressed
 * @param button The button that was pressed
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseDown(Uint8 button, unsigned x, unsigned y);
/**
 * Called when a mouse button was released
 * @param button The button that was released
 * @param x The x position where the mouse was clicked
 * @param y The y position where the mouse was clicked
 */
void onMouseUp(Uint8 button, unsigned x, unsigned y);

/**
 * Initializes SDL and OpenGL (creating context etc.)
 * @param width Width of the window
 * @param height Height of the window
 * @param color Number of bits for color
 */
SDL_Surface * initScreen(unsigned int width, unsigned int height, unsigned int color);

/**
 * Mainloop for application using sdl
 * @param screen The main SDL surface
 */
void mainLoop(SDL_Surface *screen);

/** Animation main loop
 * @param period - rough time between redraws in ms
 * @param screen The main SDL surface
 */
void mainLoop(unsigned period, SDL_Surface *screen);

#undef main

#endif
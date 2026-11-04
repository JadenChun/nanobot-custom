import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

import { floatingSurfaceElevationClassName } from "@/components/ui/floating-surface";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";

const TooltipProvider = TooltipPrimitive.Provider;
const TooltipTrigger = TooltipPrimitive.Trigger;

/**
 * Tooltip that only opens for a hovering, fine pointer.
 *
 * Radix opens on hover and on focus. A touch tap focuses the trigger, so the
 * tooltip appeared over the action it had just performed (for example above the
 * file preview opened from a chip). Touch devices get no tooltip at all; the
 * same information stays available inline or in the opened surface.
 */
function Tooltip({
  open,
  onOpenChange,
  ...props
}: React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Root>) {
  const finePointer = useMediaQuery("(hover: hover) and (pointer: fine)", true);
  return (
    <TooltipPrimitive.Root
      {...props}
      {...(finePointer
        ? { open, onOpenChange }
        : { open: false, onOpenChange: undefined })}
    />
  );
}

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        floatingSurfaceElevationClassName,
        "z-50 overflow-hidden rounded-control px-3 py-1.5 text-xs animate-in fade-in-0 zoom-in-95",
        className,
      )}
      {...props}
    />
  </TooltipPrimitive.Portal>
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

export { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger };

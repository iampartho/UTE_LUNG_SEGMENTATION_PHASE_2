import SimpleITK as sitk
def largest_object(im,n=1):
    connect = sitk.ConnectedComponent(im)
    relabel = sitk.RelabelComponent(connect)
    largest = sitk.BinaryThreshold(relabel,1,n)
    return largest

